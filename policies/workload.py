import time
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score
# 评测框架提供的接口：add(block_id), del(block_id)
# 这里默认已由评测环境注入。

@dataclass
class BlockMeta:
    block_id: int
    workload: Tuple[str, int]  # (req_type, turn)
    last_access: float         # monotonic time
    offset: float              # 该 block 在 prompt 中的相对位置，前缀越小越重要


@dataclass
class WStats:
    # 对每个 workload 维护指数分布参数估计
    mean_inter_hit: float = 30.0     # 命中间隔的 EMA（秒），默认 30s 作冷启动
    lambda_rate: float = 1.0 / 30.0  # λ = 1 / mean
    # 近未来窗口（life）长度（秒），用 P90 近似并裁剪
    life: float = 60.0               # 冷启动：60s
    # 观测次数（用于预热）
    n_obs: int = 0


class WorkloadAwareKVCachePolicy(BaseKVCachePolicy):
    """
    基于工作负载的概率淘汰（WA-ProbEvict）：
    - 估计每个 workload 的命中间隔 ~ Exp(λ)，命中时更新 EMA；
    - 淘汰时对每个 workload 仅取其 LRU 队列的队首作为候选；
    - 在候选集合上，选未来 life 内“再用概率”最小者；并以 -offset 作为次级比较维度；
    - 命中或插入时，更新 last_access，维护各 workload 的 LRU 队列顺序。
    """
    name = "wa_prob_evict"

    # ===== 可调参数 =====
    EMA_ALPHA = 0.1          # 命中间隔 EMA 衰减
    MIN_MEAN = 1e-2          # 防 0/极端：mean 下界
    MAX_MEAN = 3600.0        # mean 上界（1小时）
    MIN_LIFE = 0.5           # life 下界（s）
    MAX_LIFE = 600.0         # life 上界（10min）
    # life 采用 P90 近似：life = ln(10)/lambda，且做裁剪
    # 低观测数/未知分布时降级为 recency（LRU）+ 前缀优先

    def init(self) -> None:
        self.cache: set[int] = set()
        self.block_meta: Dict[int, BlockMeta] = {}
        # 每个 workload 一个 LRU（OrderedDict: block_id -> None），越靠前越久未使用
        self.wl_lru: Dict[Tuple[str, int], OrderedDict[int, None]] = defaultdict(OrderedDict)
        # 工作负载统计
        self.wl_stats: Dict[Tuple[str, int], WStats] = defaultdict(WStats)
        # 全局后备 LRU，供冷启动或异常时兜底
        self.global_lru: OrderedDict[int, None] = OrderedDict()

    # -------- 入口：每次 block 访问 --------
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        now = time.monotonic()
        workload = self._get_workload(meta)

        if block_id in self.cache:
            # 命中：更新间隔统计 + LRU 顺序
            self._on_hit(block_id, workload, now)
            return True

        # Miss：需要插入，可能先淘汰
        if len(self.cache) >= self.cache_size:
            victim = self._select_victim(now)
            if victim is not None:
                self._evict(victim)

        # 插入
        offset = self._compute_offset_on_miss(block_id, prompt_blocks)
        self._insert(block_id, workload, now, offset)
        return False

    # -------- 关键：选择牺牲者 --------
    def _select_victim(self, now: float) -> Optional[int]:
        # 汇总候选：每个 workload 的 LRU 队首（最久未使用）
        candidates: List[int] = []
        for w, lru in self.wl_lru.items():
            # 跳过空队列
            if not lru:
                continue
            # OrderedDict 第一个是最久未使用
            first_key = next(iter(lru))
            candidates.append(first_key)

        # 如果异常为空，退化为全局 LRU
        if not candidates:
            if self.global_lru:
                return next(iter(self.global_lru))
            return None

        # 计算每个候选的优先级：未来 life 内“再用概率”，越小越先淘汰；
        # 并以 -offset 作为第二维（offset 大＝越靠尾＝更想淘汰）
        best_block = None
        best_key = (float("inf"), float("inf"))  # (reuse_prob, -offset) —— 我们选“最小”
        for bid in candidates:
            bm = self.block_meta.get(bid)
            if bm is None:
                # 不该发生，兜底
                return bid

            ws = self.wl_stats[bm.workload]
            # 若观测不足，退化为 recency（相当于 reuse_prob=+inf，保证优先由 recency 决定）
            if ws.n_obs < 3:
                reuse_prob = float("inf")
                life = self.MIN_LIFE
            else:
                lam = max(1.0 / max(ws.mean_inter_hit, self.MIN_MEAN), 1e-6)
                # life = ln(10)/lambda（P90），再裁剪
                life = max(min(math.log(10.0) / lam, self.MAX_LIFE), self.MIN_LIFE)
                # 已空闲时长
                t_idle = max(now - bm.last_access, 0.0)
                # 未来 life 内的再用概率：e^{-λ t} (1 - e^{-λ life})
                try:
                    reuse_prob = math.exp(-lam * t_idle) * (1.0 - math.exp(-lam * life))
                except OverflowError:
                    reuse_prob = 0.0

            key = (reuse_prob, -bm.offset)
            if key < best_key:
                best_key = key
                best_block = bid

        return best_block

    # -------- 命中时：更新统计与 LRU --------
    def _on_hit(self, block_id: int, new_workload: Tuple[str, int], now: float) -> None:
        bm = self.block_meta[block_id]
        # 统计：命中间隔
        delta = max(now - bm.last_access, 0.0)
        ws = self.wl_stats[new_workload]
        old_mean = ws.mean_inter_hit
        # EMA 更新
        new_mean = (1.0 - self.EMA_ALPHA) * old_mean + self.EMA_ALPHA * min(delta, self.MAX_MEAN)
        ws.mean_inter_hit = max(min(new_mean, self.MAX_MEAN), self.MIN_MEAN)
        ws.lambda_rate = 1.0 / ws.mean_inter_hit
        # life 更新：P90 截断
        ws.life = max(min(math.log(10.0) / ws.lambda_rate, self.MAX_LIFE), self.MIN_LIFE)
        ws.n_obs += 1

        # 如果 workload 变了（跨类型/跨轮次命中），迁移到新 workload 的 LRU
        if bm.workload != new_workload:
            # 从旧 workload LRU 移除
            try:
                del self.wl_lru[bm.workload][block_id]
                if not self.wl_lru[bm.workload]:
                    self.wl_lru.pop(bm.workload, None)
            except KeyError:
                pass
            # 加入新 workload 的 LRU 末尾（最新）
            self.wl_lru[new_workload][block_id] = None

            bm.workload = new_workload

        # 刷新 last_access
        bm.last_access = now

        # LRU：把该块移到其 workload LRU 的末尾（最新）
        try:
            lru = self.wl_lru[bm.workload]
            if block_id in lru:
                lru.move_to_end(block_id)
            else:
                lru[block_id] = None
        except KeyError:
            self.wl_lru[bm.workload] = OrderedDict({block_id: None})

        # 全局 LRU 同步
        if block_id in self.global_lru:
            self.global_lru.move_to_end(block_id)
        else:
            self.global_lru[block_id] = None

    # -------- 插入与淘汰 --------
    def _insert(self, block_id: int, workload: Tuple[str, int], now: float, offset: float) -> None:
        self.cache.add(block_id)
        self.block_meta[block_id] = BlockMeta(
            block_id=block_id, workload=workload, last_access=now, offset=offset
        )
        # 插入对应 workload LRU 尾部 & 全局 LRU 尾部
        self.wl_lru[workload][block_id] = None
        self.global_lru[block_id] = None

    def _evict(self, block_id: int) -> None:
        if block_id not in self.cache:
            return
        bm = self.block_meta.pop(block_id, None)
        self.cache.remove(block_id)
        self.global_lru.pop(block_id, None)
        if bm is not None:
            lru = self.wl_lru.get(bm.workload)
            if lru is not None:
                lru.pop(block_id, None)
                if not lru:
                    self.wl_lru.pop(bm.workload, None)

    # -------- 工具：取 workload / 计算 offset --------
    def _get_workload(self, meta: Optional[Dict[str, Any]]) -> Tuple[str, int]:
        """
        从 meta 提取 (req_type, turn)。都缺省时使用 ('unknown', 1)。
        - req_type: e.g. 'api' / 'text' / 'file' / 'search' / 'multimodal' / 'agent' ...
        - turn: 第几轮（1 表示单轮）
        """
        if not meta:
            return ("unknown", 1)
        # 兼容多种键名
        req_type = (
            meta.get("req_type")
            or meta.get("type")
            or meta.get("app_type")
            or "unknown"
        )
        turn = int(meta.get("turn", meta.get("round", 1)) or 1)
        # 归一化
        if not isinstance(req_type, str):
            req_type = str(req_type)
        if turn < 1:
            turn = 1
        return (req_type, turn)

    def _compute_offset_on_miss(
        self, block_id: int, prompt_blocks: Iterable[int]
    ) -> float:
        """
        仅在 miss 时计算 offset，避免每次访问都 O(n)。
        offset ∈ [0,1]，越小越靠前缀；没找到则置 1.0（保守当作尾部）。
        """
        try:
            if prompt_blocks is None:
                return 1.0
            seq = list(prompt_blocks)
            n = len(seq)
            if n == 0:
                return 1.0
            # 找到 block 的位置（若不在本次 prompt 中，说明是复用历史块，给个偏尾默认值）
            try:
                idx = seq.index(block_id)
                return idx / max(n - 1, 1)
            except ValueError:
                return 0.75  # 不在 prompt 中，默认偏尾
        except Exception:
            return 1.0
