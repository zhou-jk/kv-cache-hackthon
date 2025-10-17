from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterable, Set, Tuple
from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score

class WorkloadAwareKVCachePolicy(BaseKVCachePolicy):
    """
    WA-Prob（高效版）: 工作负载感知的复用概率驱逐策略
    - 概念对齐论文：按“类别”拟合指数分布，驱逐时用 ReuseProb 排序 + 空间局部性（offset 越小越重要）
    - 时间复杂度：命中/插入 O(1)，驱逐 O(W)（W=活跃类别数）
    - 需要 meta: {"category": <hashable>, "offset": <int, 0为最靠前>, "life_hint": <可选float>}
    """

    name = "wa_prob_evict"

    # ---------- 可调参数 ----------
    # EMA 平滑系数（越大越保守）
    LAMBDA_EMA_ALPHA: float = 0.9     # 用于复用间隔的 EMA（再转成 lambda=1/mu）
    LIFE_EMA_ALPHA: float   = 0.9     # 类别寿命窗口的 EMA
    # 初始缺省值（步数尺度）：没有统计时退化为 LRU
    DEFAULT_MEAN_REUSE: float = 128.0  # 缺省平均复用间隔 mu
    DEFAULT_LIFE: float       = 64.0   # 缺省寿命窗口 life
    # 达到这个样本数后才认为“lambda 估计有效”
    MIN_SAMPLES_FOR_PROB: int = 8

    def init(self) -> None:
        # 基本集合
        self.cache: Set[int] = set()
        self.step: int = 0

        # 统计：每块
        self.freq: Dict[int, int]        = defaultdict(int)
        self.last_access: Dict[int, int] = {}            # 上次访问步
        self.block_cat: Dict[int, Any]   = {}            # 块 -> 类别
        self.block_off: Dict[int, int]   = {}            # 块 -> offset（越小越靠前）

        # 按类别的 LRU 队列：cat -> OrderedDict[block_id]，头部是最老
        self.cat_queues: Dict[Any, OrderedDict] = defaultdict(OrderedDict)

        # 按类别的概率模型参数（在线估计）
        self.cat_mu: Dict[Any, float]      = defaultdict(lambda: self.DEFAULT_MEAN_REUSE)   # 复用间隔均值
        self.cat_lambda: Dict[Any, float]  = defaultdict(lambda: 1.0 / self.DEFAULT_MEAN_REUSE)
        self.cat_life: Dict[Any, float]    = defaultdict(lambda: self.DEFAULT_LIFE)
        self.cat_samples: Dict[Any, int]   = defaultdict(int)  # 命中样本数（间隔数）

    # ------------------------ 工具函数 ------------------------
    @staticmethod
    def _call_backend(fn_name: str, block_id: int) -> None:
        fn = globals().get(fn_name, None)
        if callable(fn):
            try:
                fn(block_id)
            except Exception:
                pass

    def _backend_add(self, block_id: int) -> None:
        self._call_backend("add", block_id)

    def _backend_del(self, block_id: int) -> None:
        for name in ("delete", "evict", "remove"):
            fn = globals().get(name, None)
            if callable(fn):
                try:
                    fn(block_id)
                    return
                except Exception:
                    pass
        self._call_backend("del", block_id)

    @staticmethod
    def _reuse_prob(lambda_w: float, t_since_last: float, life_w: float) -> float:
        """
        指数分布的区间复用概率：
        P = exp(-λ t) * (1 - exp(-λ * life))
        数值稳定性：当 λ 很小 -> 近似 (life / (t + large))；但这里 λ 来源于 mu=1/λ，已做下限保护。
        """
        # 基本保护，避免 underflow/overflow
        if lambda_w <= 0.0:
            return 0.0
        # e^{-λ t}:
        from math import exp
        a = exp(-lambda_w * max(t_since_last, 0.0))
        # (1 - e^{-λ life})
        b = 1.0 - exp(-lambda_w * max(life_w, 0.0))
        p = a * b
        # 数值夹紧
        if p < 0.0: p = 0.0
        if p > 1.0: p = 1.0
        return p

    def _update_category_stats_on_hit(self, cat: Any, dt: int, life_hint: float | None) -> None:
        """
        命中时，使用“距离上次访问的步数 dt”更新该类别的 mu / lambda；
        若提供 life_hint，则同时更新类别寿命窗口。
        """
        # 更新 mu 的 EMA
        mu_old = self.cat_mu[cat]
        mu_new = self.LAMBDA_EMA_ALPHA * mu_old + (1.0 - self.LAMBDA_EMA_ALPHA) * float(max(dt, 1))
        self.cat_mu[cat] = mu_new

        # lambda = 1 / mu
        self.cat_lambda[cat] = 1.0 / max(mu_new, 1e-6)
        self.cat_samples[cat] += 1

        # 更新 life（可选）
        if life_hint is not None:
            life_old = self.cat_life[cat]
            life_new = self.LIFE_EMA_ALPHA * life_old + (1.0 - self.LIFE_EMA_ALPHA) * float(max(life_hint, 0.0))
            # 基本保护：life 不应太小
            self.cat_life[cat] = max(life_new, 1.0)

    # ------------------------ 受害者选择 ------------------------
    def _pick_victim(self) -> int:
        """
        候选 = 每个活跃类别的“最老块”（各自 OrderedDict 的第一个）
        计算其区间复用概率 P_w(t, life)，选择 P 最小者；若相等，驱逐 offset 更大的；再相等，比 t 大的。
        复杂度 O(W)
        """
        best: Tuple[float, int, int, int] | None = None
        # 元组： (prob, -offset, t_since_last, block_id)  # 注意 -offset：offset 大(不前缀)优先驱逐
        # 取负号的原因：我们做“min”，希望 offset 大者被驱逐，等价于比较 (-offset) 更小
        for cat, q in self.cat_queues.items():
            if not q:
                continue
            # 类别最老块：OrderedDict 的第一个
            oldest_block = next(iter(q))                # O(1)
            t_since_last = self.step - self.last_access.get(oldest_block, self.step)
            lambda_w = self.cat_lambda[cat]
            life_w = self.cat_life[cat]
            has_enough = self.cat_samples[cat] >= self.MIN_SAMPLES_FOR_PROB

            if has_enough:
                prob = self._reuse_prob(lambda_w, float(t_since_last), float(life_w))
                # 空间局部性：offset 越小越重要 -> 在“同概率”时尽量保留小 offset
                off = self.block_off.get(oldest_block, 1 << 30)
                key = (prob, -int(off), int(t_since_last), int(oldest_block))
            else:
                # 采样不足 -> 退化为 LRU + 空间局部性（概率视为 None/同等）
                # 我们用一个固定 prob 值（例如 0.5）使其只受 (-offset, t) 排序
                prob = 0.5
                off = self.block_off.get(oldest_block, 1 << 30)
                key = (prob, -int(off), int(t_since_last), int(oldest_block))

            if (best is None) or (key < best):
                best = key

        # 兜底：若 best 仍为空（理论上不会），从全局任取一个
        if best is None:
            # 随便拿到某个类别的最老块
            for q in self.cat_queues.values():
                if q:
                    return next(iter(q))
            # 再兜底：从 cache 里 pop 一个
            return next(iter(self.cache))

        # 返回选中的 block_id
        return best[-1]

    # ------------------------ 主入口 ------------------------
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        """
        meta 约定：
          - category: 任意可哈希（str/int），默认 "default"
          - offset:   int，越小越靠前；缺省视为很大
          - life_hint: float，可选；若提供则用于更新该类别寿命 EMA
        """
        self.step += 1

        if meta is None:
            meta = {}
        cat = meta.get("category", "default")
        off = int(meta.get("offset", 1 << 30))
        life_hint = meta.get("life_hint", None)
        if life_hint is not None:
            try:
                life_hint = float(life_hint)
            except Exception:
                life_hint = None

        # ---------- 命中 ----------
        if block_id in self.cache:
            # 复用间隔 = 当前步 - 上次访问步
            prev = self.last_access.get(block_id, self.step)
            dt = max(self.step - prev, 1)

            # 更新全局/块统计
            self.freq[block_id] += 1
            self.last_access[block_id] = self.step

            # 类别 LRU 队列移动到末尾（最新）
            q = self.cat_queues[self.block_cat[block_id]]
            # OrderedDict 的“刷新”方式：先删再插或使用 move_to_end
            if block_id in q:
                q.move_to_end(block_id, last=True)

            # 在线更新类别的概率模型
            self._update_category_stats_on_hit(cat=self.block_cat[block_id], dt=dt, life_hint=life_hint)

            # 更新 offset（同一块在不同请求里 offset 可能略变；取更小的有利于保留）
            old_off = self.block_off.get(block_id, off)
            if off < old_off:
                self.block_off[block_id] = off

            return True

        # ---------- 未命中：必要时驱逐 ----------
        if len(self.cache) >= self.cache_size > 0:
            victim = self._pick_victim()

            # 从全局集合移除
            if victim in self.cache:
                self.cache.remove(victim)

            # 从类别队列移除
            vcat = self.block_cat.get(victim, "default")
            q = self.cat_queues[vcat]
            if victim in q:
                q.pop(victim, None)

            # 清理映射
            self.block_cat.pop(victim, None)
            self.block_off.pop(victim, None)
            self.last_access.pop(victim, None)
            # freq 可选：保留可做“短期频率”统计，不影响正确性
            # self.freq.pop(victim, None)

            # 后端回调
            self._backend_del(victim)

        # ---------- 插入 ----------
        self.cache.add(block_id)
        self.block_cat[block_id] = cat
        # offset：若之前见过同块，保留更小的 offset（更前缀）
        old_off = self.block_off.get(block_id, off)
        self.block_off[block_id] = min(old_off, off)

        # 类别队列：放到队尾（最新）
        q = self.cat_queues[cat]
        q[block_id] = None

        # 基本统计
        self.freq[block_id] += 1
        self.last_access[block_id] = self.step

        # 如有 life_hint，可用来“提前”校准该类寿命
        if life_hint is not None:
            life_old = self.cat_life[cat]
            life_new = self.LIFE_EMA_ALPHA * life_old + (1.0 - self.LIFE_EMA_ALPHA) * float(max(life_hint, 0.0))
            self.cat_life[cat] = max(life_new, 1.0)

        # 后端回调
        self._backend_add(block_id)

        return False
