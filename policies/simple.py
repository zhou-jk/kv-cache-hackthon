from __future__ import annotations

from typing import Dict, Iterable, List
from collections import OrderedDict, defaultdict

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score
from sortedcontainers import SortedList
from heapdict import heapdict

class FIFOKVCachePolicy(BaseKVCachePolicy):
    name = "fifo"
    miss: int = 0
    access_count: int = 0

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
        self.access_count += 1
        if block_id in self.cache:
            return True
        if len(self.cache) >= self.cache_size:
            victim = self.queue.pop(0)
            self.cache.remove(victim)
        self.cache.add(block_id)
        self.queue.append(block_id)
        self.miss += 1
        return False


class LRUKVCachePolicy(BaseKVCachePolicy):
    name = "lru"
    miss: int = 0
    access_count: int = 0

    def init(self) -> None:
        from collections import OrderedDict

        self.table: OrderedDict[int, None] = OrderedDict()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict | None = None,
    ) -> bool:
        self.access_count += 1
        del prompt_blocks, meta
        if block_id in self.table:
            self.table.move_to_end(block_id)
            return True
        if len(self.table) >= self.cache_size:
            self.table.popitem(last=False)
        self.table[block_id] = None
        self.miss += 1
        return False

import heapq
from collections import defaultdict
from typing import Any, Dict, Iterable, Set

class PrefixAwareKVCachePolicy(BaseKVCachePolicy):
    """
    PALFU (优化版): Prefix-Aware LFU/LRU Fusion Policy
    - 关键优化：两堆 + 懒失效 + 增量维护前缀
      * pin_heap: 维护 r 个最小编号（max-heap by id）
      * cand_heap: 维护非前缀受害者候选（min-heap by score）
      * all_id_minheap: 所有 id 的 min-heap（用于快速补足前缀）
    - 复杂度：驱逐/插入/命中更新 ~ O(log N)
    """
    name = "palfu"

    # ------------------------ 可调参数 ------------------------
    PREFIX_RESERVE_RATIO: float = 0.4
    PREFIX_EMA_ALPHA: float = 0.9
    W_FREQ: float = 1.0
    W_RECENCY: float = 0.6
    W_IDBIAS: float = 1.4
    STATS_GC_MULTIPLIER: int = 1000

    def init(self) -> None:
        self.cache: Set[int] = set()
        self.cache_size: int = getattr(self, "cache_size", 0)  # 上层应提供
        self.step: int = 0

        # 统计
        self.freq: Dict[int, int] = defaultdict(int)
        self.last_access: Dict[int, int] = {}

        # 前缀 EMA / 指纹
        self.prefix_len_ema: float = 0.0
        self._last_prompt_fp: int | None = None

        # ---------- 堆 & 懒失效状态 ----------
        # 前缀的“r 个最小 id” -> max-heap: 存 (-id, id) 便于取出编号最大的前缀
        self._pin_heap: list[tuple[int, int]] = []
        self._is_pinned: Set[int] = set()

        # 非前缀候选（按 score） -> min-heap: (score, gen, id)
        self._cand_heap: list[tuple[float, int, int]] = []
        self._stamp: Dict[int, int] = {}   # id -> 最新代数（用于堆条目懒失效）

        # 全部 id 的最小堆（用于扩张前缀时快速拿到最小未 pinned 的 id）
        self._all_id_minheap: list[int] = []

        # 版本计数器（用于 cand_heap 懒失效）
        self._gen: int = 0

        # 缓存 id->1/id，省一点重复计算
        self._inv_id_cache: Dict[int, float] = {}

        # 当前目标前缀数 r
        self._r_target: int = 0

    # ------------------------ 后端 add/del ------------------------
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

    # ------------------------ 前缀 EMA ------------------------
    @staticmethod
    def _contiguous_prefix_len(blocks: Iterable[int]) -> int:
        s = set(blocks)
        if 1 not in s:
            return 0
        k = 1
        while (k + 1) in s:
            k += 1
        return k

    def _maybe_update_prefix_ema(self, prompt_blocks: Iterable[int]) -> None:
        if not prompt_blocks:
            return
        bl = list(prompt_blocks)
        mn, mx, ln = min(bl), max(bl), len(bl)
        fp = (mx << 21) ^ (mn << 11) ^ ln
        if self._last_prompt_fp == fp:
            return
        self._last_prompt_fp = fp

        prefix_len = self._contiguous_prefix_len(bl)
        self.prefix_len_ema = (
            self.PREFIX_EMA_ALPHA * self.prefix_len_ema
            + (1.0 - self.PREFIX_EMA_ALPHA) * float(prefix_len)
        )

        # r 可能变化，做一次增量再平衡
        self._rebalance_pinned_if_needed()

    def _pinned_target_count(self) -> int:
        hard_cap = int(self.cache_size * self.PREFIX_RESERVE_RATIO)
        ema_cap = int(self.prefix_len_ema + 1e-6)
        return max(0, min(hard_cap, ema_cap))

    # ------------------------ 分值 ------------------------
    def _score(self, block_id: int) -> float:
        f = float(self.freq.get(block_id, 0))
        la = self.last_access.get(block_id, 0)
        rec = 1.0 / (1.0 + (self.step - la))
        inv_id = self._inv_id_cache.get(block_id)
        if inv_id is None:
            inv_id = 1.0 / float(block_id) if block_id > 0 else 1.0
            self._inv_id_cache[block_id] = inv_id
        return self.W_FREQ * f + self.W_RECENCY * rec + self.W_IDBIAS * inv_id

    # ------------------------ 堆维护工具 ------------------------
    def _rebalance_pinned_if_needed(self) -> None:
        """让前缀的大小匹配最新 r，仅做增量调整（O(|Δr| log N))."""
        new_r = self._pinned_target_count()
        if new_r == self._r_target:
            return

        if new_r < self._r_target:
            # 缩小：弹出若干“编号最大”的前缀转入候选堆
            k = self._r_target - new_r
            for _ in range(k):
                # 从 max-heap 取出最大的 id
                while self._pin_heap:
                    negid, bid = heapq.heappop(self._pin_heap)
                    if bid in self.cache and bid in self._is_pinned:
                        self._is_pinned.remove(bid)
                        # 进 cand 堆
                        self._push_candidate(bid)
                        break
            self._r_target = new_r
        else:
            # 扩大：把“最小的未 pinned id”补进前缀
            k = new_r - self._r_target
            for _ in range(k):
                nid = self._pop_smallest_unpinned_id()
                if nid is None:
                    break
                self._pin(nid)
            self._r_target = new_r

    def _pin(self, bid: int) -> None:
        """把某个 id 标为前缀，进 pin_heap（max-heap by id）"""
        if bid in self._is_pinned:
            return
        self._is_pinned.add(bid)
        heapq.heappush(self._pin_heap, (-bid, bid))

    def _push_candidate(self, bid: int) -> None:
        """把非前缀的块压到候选堆（按 score），并递增代数用于懒失效。"""
        if bid in self._is_pinned or bid not in self.cache:
            return
        self._gen += 1
        self._stamp[bid] = self._gen
        s = self._score(bid)
        heapq.heappush(self._cand_heap, (s, self._gen, bid))

    def _pop_smallest_unpinned_id(self) -> int | None:
        """从 all_id_minheap 里取一个当前存在且未 pinned 的最小 id。"""
        while self._all_id_minheap:
            bid = heapq.heappop(self._all_id_minheap)
            if bid in self.cache and bid not in self._is_pinned:
                return bid
        return None

    def _pick_victim(self) -> int:
        """优先在非前缀里选分值最低 (O(log N))；若为空，再从前缀里选编号最大。"""
        if not self.cache:
            raise RuntimeError("cache empty when picking victim")

        # 非前缀候选
        while self._cand_heap:
            s, gen, bid = heapq.heappop(self._cand_heap)
            # 懒失效：淘汰旧项 / 已转为前缀 / 已被驱逐
            if (bid not in self.cache) or (bid in self._is_pinned) or (self._stamp.get(bid) != gen):
                continue
            return bid

        # 非前缀空了 -> 退化到从前缀里挑编号最大
        while self._pin_heap:
            negid, bid = heapq.heappop(self._pin_heap)
            if bid in self.cache and bid in self._is_pinned:
                self._is_pinned.remove(bid)
                return bid

        # 理论上到不了
        return max(self.cache)

    def _admit_new_block(self, bid: int) -> None:
        """新块加入：依据当前 r 与前缀最大编号，O(log N) 决定是否进前缀。"""
        heapq.heappush(self._all_id_minheap, bid)

        # 确保 r 是最新
        self._rebalance_pinned_if_needed()

        if len(self._is_pinned) < self._r_target:
            self._pin(bid)
            return

        # 看看是否比当前“前缀中的最大编号”还小，是则交换
        if self._pin_heap:
            cur_max_negid, cur_max = self._pin_heap[0]
            cur_max = cur_max  # top 是编号最大的前缀
            if bid < cur_max:
                # 最大的前缀下放为候选
                heapq.heappop(self._pin_heap)
                if cur_max in self._is_pinned:
                    self._is_pinned.remove(cur_max)
                    self._push_candidate(cur_max)
                # 新块进入前缀
                self._pin(bid)
                return

        # 否则作为非前缀候选进入 cand_heap
        self._push_candidate(bid)

    # ------------------------ 统计 GC ------------------------
    def _maybe_gc_stats(self) -> None:
        limit = self.STATS_GC_MULTIPLIER * max(1, self.cache_size)
        if len(self.freq) <= limit and len(self.last_access) <= limit:
            return

        keep: Set[int] = set(self.cache)
        # 高频
        if len(keep) < limit // 2:
            for k, _ in sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)[: (limit // 2)]:
                keep.add(k)
        # 最近
        if len(keep) < limit:
            for k, _ in sorted(self.last_access.items(), key=lambda kv: kv[1], reverse=True)[: (limit - len(keep))]:
                keep.add(k)

        for d in (self.freq, self.last_access, self._stamp, self._inv_id_cache):
            for k in list(d.keys()):
                if k not in keep:
                    d.pop(k, None)

    # ------------------------ 主流程 ------------------------
    def access(self, block_id: int, prompt_blocks: Iterable[int], meta: Dict[str, Any] | None = None) -> bool:
        self.step += 1
        self.access_count += 1
        try:
            self._maybe_update_prefix_ema(prompt_blocks)
        except Exception:
            pass

        if block_id in self.cache:
            # 命中：更新统计；若非前缀，刷新候选分数（懒失效）
            self.freq[block_id] += 1
            self.last_access[block_id] = self.step
            if block_id not in self._is_pinned:
                self._push_candidate(block_id)
            # 统计过大时再收缩
            if len(self.freq) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size) or \
               len(self.last_access) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size):
                self._maybe_gc_stats()
            return True

        # 未命中：必要时驱逐
        if len(self.cache) >= self.cache_size > 0:
            victim = self._pick_victim()
            if victim in self.cache:
                self.cache.remove(victim)
                # 受害者可能来自前缀，也可能来自 cand；两边状态都懒失效，不需额外清理
                self._backend_del(victim)

        # 插入新块
        self.cache.add(block_id)
        self.miss += 1
        self._backend_add(block_id)
        self.freq[block_id] += 1
        self.last_access[block_id] = self.step

        self._admit_new_block(block_id)

        # 超限时再 GC 一下（按容量触发，而非固定步长）
        if len(self.freq) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size) or \
           len(self.last_access) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size):
            self._maybe_gc_stats()

        return False
