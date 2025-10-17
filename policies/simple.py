from __future__ import annotations

from typing import Dict, Iterable, List
from collections import OrderedDict, defaultdict

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score


class FIFOKVCachePolicy(BaseKVCachePolicy):
    name = "fifo"

    def init(self) -> None:
        self.queue: List[int] = []
        self.cache: set[int] = set()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict | None = None,
    ) -> bool:
        del prompt_blocks, meta
        if block_id in self.cache:
            return True
        if len(self.cache) >= self.cache_size:
            victim = self.queue.pop(0)
            self.cache.remove(victim)
        self.cache.add(block_id)
        self.queue.append(block_id)
        return False


class LRUKVCachePolicy(BaseKVCachePolicy):
    name = "lru"

    def init(self) -> None:
        from collections import OrderedDict

        self.table: OrderedDict[int, None] = OrderedDict()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict | None = None,
    ) -> bool:
        del prompt_blocks, meta
        if block_id in self.table:
            self.table.move_to_end(block_id)
            return True
        if len(self.table) >= self.cache_size:
            self.table.popitem(last=False)
        self.table[block_id] = None
        return False


class PrefixAwareKVCachePolicy(BaseKVCachePolicy):
    """
    PALFU: Prefix-Aware LFU/LRU Fusion Policy
    - Prefill-only, block-level KV cache policy.
    - Core: prefix pinning + (frequency + recency + id-bias) score + 2-stage eviction.
    """
    name = "palfu"

    # ------------------------ 可调参数 ------------------------
    # 预留给“前缀”的比例上限（占总 cache 的份额），0.0~0.9 之间较合理
    PREFIX_RESERVE_RATIO: float = 0.4
    # 前缀长度 EMA 的平滑系数（值越大越保守）
    PREFIX_EMA_ALPHA: float = 0.9
    # 分值权重：频率、时间新近、编号偏置
    W_FREQ: float = 1.0
    W_RECENCY: float = 0.6
    W_IDBIAS: float = 1.4
    # 统计裁剪阈值（避免 dict 无限增长）
    STATS_GC_MULTIPLIER: int = 8  # 频率/时间字典上限 ~ 8 * cache_size

    def init(self) -> None:
        self.cache: Set[int] = set()
        self.step: int = 0                          # 全局时间步（每次 access 自增）
        self.freq: Dict[int, int] = defaultdict(int)  # 块全局访问次数（LFU）
        self.last_access: Dict[int, int] = {}         # 最近访问时间步（LRU)
        # 自适应前缀估计（EMA）
        self.prefix_len_ema: float = 0.0
        # 最近一次用于计算前缀的 prompt 指纹（避免重复计算）
        self._last_prompt_fp: int | None = None

    # ------------------------ 对接底层add/del（若存在） ------------------------
    @staticmethod
    def _call_backend(fn_name: str, block_id: int) -> None:
        fn = globals().get(fn_name, None)
        if callable(fn):
            try:
                fn(block_id)
            except Exception:
                pass

    def _backend_add(self, block_id: int) -> None:
        # 常见实现叫 add()
        self._call_backend("add", block_id)

    def _backend_del(self, block_id: int) -> None:
        # 有的环境叫 delete/evict/remove，这里都尝试一下
        for name in ("delete", "evict", "remove"):
            fn = globals().get(name, None)
            if callable(fn):
                try:
                    fn(block_id)
                    return
                except Exception:
                    pass
        # 若运行环境恰好提供了名为 del 的函数（极少见），也做保护性尝试
        self._call_backend("del", block_id)

    # ------------------------ 前缀长度估计 ------------------------
    @staticmethod
    def _contiguous_prefix_len(blocks: Iterable[int]) -> int:
        """
        计算从 1 开始的“连续前缀长度”：
        例如 {1,2,3,5,8} -> 3；{2,3,4} -> 0；{1,2,6} -> 2
        """
        s = set(blocks)
        if 1 not in s:
            return 0
        k = 1
        # 在端侧/题目 trace 中，编号一般不离谱；逐步判断足够快
        while (k + 1) in s:
            k += 1
        return k

    def _maybe_update_prefix_ema(self, prompt_blocks: Iterable[int]) -> None:
        """
        只在一个 prompt 的“第一块访问”附近更新一次前缀 EMA，避免反复扫描。
        粗略判定：当访问的 block_id 等于该 prompt 的最小编号（通常是 1）时更新。
        """
        if not prompt_blocks:
            return
        # 指纹（不可逆快速 hash）：max_id ^ len ^ min_id
        blist = list(prompt_blocks)
        mn, mx, ln = min(blist), max(blist), len(blist)
        fp = (mx << 21) ^ (mn << 11) ^ ln
        if self._last_prompt_fp == fp:
            return
        self._last_prompt_fp = fp

        if mn == min(blist):  # “看起来像是新 prompt 的开始”
            prefix_len = self._contiguous_prefix_len(blist)
            self.prefix_len_ema = (
                self.PREFIX_EMA_ALPHA * self.prefix_len_ema
                + (1.0 - self.PREFIX_EMA_ALPHA) * float(prefix_len)
            )

    def _pinned_target_count(self) -> int:
        """
        需要预留给前缀的目标数量：受 cache_size * ratio 与 prefix_ema 双重约束。
        """
        hard_cap = int(self.cache_size * self.PREFIX_RESERVE_RATIO)
        ema_cap = int(self.prefix_len_ema + 1e-6)
        return max(0, min(hard_cap, ema_cap))

    def _current_pinned_set(self) -> Set[int]:
        """
        依据当前 cache 内容与目标前缀长度，确定“本轮不希望驱逐”的 pinned 集合。
        这里的启发式很简单：把 cache 里编号最小的 r 个当作前缀保留（r 为目标数）。
        """
        r = self._pinned_target_count()
        if r <= 0 or not self.cache:
            return set()
        smallest_r = sorted(self.cache)[:min(r, len(self.cache))]
        return set(smallest_r)

    # ------------------------ 分值/受害者选择 ------------------------
    def _score(self, block_id: int) -> float:
        """
        越大越重要（不该被驱逐）：
          score = Wf * freq
                + Wr * recency
                + Wi * id_bias       (小编号更大)
        其中 recency = 1 / (1 + step - last_access)
            id_bias = 1 / block_id
        """
        f = float(self.freq.get(block_id, 0))
        la = self.last_access.get(block_id, 0)
        rec = 1.0 / (1.0 + (self.step - la))  # 越近越大
        id_bias = 1.0 / float(block_id) if block_id > 0 else 1.0
        return self.W_FREQ * f + self.W_RECENCY * rec + self.W_IDBIAS * id_bias

    def _pick_victim(self) -> int:
        """
        两阶段挑选牺牲者：
        1) 先在 非前缀集合 里选分值最低者；
        2) 若非前缀为空（全是前缀），选前缀里编号最大的（尽量不破坏小编号前缀）。
        """
        if not self.cache:
            raise RuntimeError("cache empty when picking victim")

        pinned = self._current_pinned_set()
        non_prefix = [bid for bid in self.cache if bid not in pinned]

        if non_prefix:
            # 在非前缀里选“最不重要”的
            victim = min(non_prefix, key=self._score)
            return victim
        else:
            # 万不得已：从前缀里选编号最大（对前缀破坏最小）
            victim = max(self.cache)
            return victim

    # ------------------------ 统计裁剪（控制额外内存） ------------------------
    def _maybe_gc_stats(self) -> None:
        limit = self.STATS_GC_MULTIPLIER * max(1, self.cache_size)
        if len(self.freq) <= limit and len(self.last_access) <= limit:
            return
        # 只保留“当前在 cache 里”或“最近/频繁”的少量条目
        keep: Set[int] = set(self.cache)
        # 补一些最高频/最近访问项
        if len(keep) < limit // 2:
            top_by_freq = sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)[: (limit // 2)]
            keep.update(k for k, _ in top_by_freq)
        if len(keep) < limit:
            top_by_rec = sorted(self.last_access.items(), key=lambda kv: kv[1], reverse=True)[: (limit - len(keep))]
            keep.update(k for k, _ in top_by_rec)

        # 清理
        for d in (self.freq, self.last_access):
            for k in list(d.keys()):
                if k not in keep:
                    d.pop(k, None)

    # ------------------------ 主入口 ------------------------
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        # 步进时间 &（在“一个 prompt 开始”时）更新前缀 EMA
        self.step += 1
        try:
            self._maybe_update_prefix_ema(prompt_blocks)
        except Exception:
            # 避免解析 prompt_blocks 异常影响主逻辑
            pass

        # 命中：只更新统计
        if block_id in self.cache:
            self.freq[block_id] += 1
            self.last_access[block_id] = self.step
            return True

        # 未命中：如必要先驱逐
        if len(self.cache) >= self.cache_size:
            victim = self._pick_victim()
            self.cache.remove(victim)
            self._backend_del(victim)

        # 插入新块
        self.cache.add(block_id)
        self._backend_add(block_id)
        self.freq[block_id] += 1
        self.last_access[block_id] = self.step

        # 偶尔做一次统计裁剪，控制内存
        if (self.step & 0xFF) == 0:  # 每 256 次访问做一次
            self._maybe_gc_stats()

        return False