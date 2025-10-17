from __future__ import annotations

from typing import Dict, Iterable, List
from collections import OrderedDict, defaultdict

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score
from sortedcontainers import SortedList
from heapdict import heapdict
import math

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
import math
from collections import defaultdict
from typing import Iterable, Dict, Any, Optional, Set, Tuple, List

class PrefixAwareKVCachePolicy(BaseKVCachePolicy):
    """
    PALFU (优化版 with Future-Aware Weights):
    - 保留两堆 + 懒失效 + 前缀EMA框架
      * pin_heap: 维护 r 个“前缀位”的集合（仍用编号近似，以与你现有逻辑兼容）
      * cand_heap: 非前缀候选的小根堆 (keep_score 越小越先被淘汰)
      * all_id_minheap: 便于扩张前缀
    - 新增: 频率 + 新近度 + 位置 的加权 keep_score；若“本轮当前位置之后还会出现”则强保护
    - 冷启动前 X 次：退化为 LRU（但仍避开“将来要用”的块）
    - 支持 meta['type'] 影响权重（如 chat/agent 偏位置，search/RAG 偏频率）
    """
    name = "palfu"

    # ------------------------ 可调参数 ------------------------
    PREFIX_RESERVE_RATIO: float = 0.40
    PREFIX_MAX_LENGTH: int = 512
    PREFIX_EMA_ALPHA: float = 0.90

    # 加权（可被 type 覆盖）
    W_FREQ: float = 1.0
    W_RECENCY: float = 0.6
    W_POS: float = 1.2

    # 若本轮之后还会再出现 -> keep_score *= FUTURE_MULT (强保护)
    FUTURE_MULT: float = 1e6

    # 位置“头部”阈值：越靠前越像公共前缀，越该保留
    POS_HEAD_CUTOFF: int = 256

    # 冷启动与低频（前 X 次/单块频次 <= FREQ_COLD 时偏向 LRU）
    WARMUP_GLOBAL: int = 2000
    FREQ_COLD: int = 1

    # 懒失效重算容差
    HEAP_EPS: float = 1e-9

    # 统计 GC
    STATS_GC_MULTIPLIER: int = 1000

    # ------------------------ 初始化 ------------------------
    def init(self) -> None:
        self.cache: Set[int] = set()
        self.cache_size: int = getattr(self, "cache_size", 0)
        self.step: int = 0
        self.miss: int = 0
        self.access_count: int = 0

        # 统计
        self.freq: Dict[int, int] = defaultdict(int)
        self.last_access: Dict[int, int] = {}

        # 前缀 EMA / 指纹（与你原逻辑一致）
        self.prefix_len_ema: float = 0.0
        self._last_prompt_fp: Optional[int] = None

        # ---------- 堆 & 懒失效状态 ----------
        self._pin_heap: List[Tuple[int, int]] = []           # (-id, id)
        self._is_pinned: Set[int] = set()

        self._cand_heap: List[Tuple[float, int, int]] = []   # (keep_score, gen, id)
        self._stamp: Dict[int, int] = {}                     # id -> 最新代数
        self._gen: int = 0

        self._all_id_minheap: List[int] = []                 # 所有 id 的 min-heap
        self._r_target: int = 0

        # ---- 位置/未来 ----
        self._cur_prompt_key: Optional[int] = None
        self._pos_map: Dict[int, int] = {}     # 当前 prompt: id -> 位置(0-based)
        self._cur_idx: Optional[int] = None    # 当前访问位置
        self._last_meta: Optional[Dict[str, Any]] = None

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

    # ------------------------ 前缀 EMA（保留原逻辑） ------------------------
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
        bl = list(prompt_blocks)  # 需保持顺序
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
        self._rebalance_pinned_if_needed()

    # ------------------------ 位置/未来 ------------------------
    def _ensure_pos_map(self, prompt_blocks: Iterable[int]) -> None:
        seq = list(prompt_blocks)  # 必须有序；若传 set 会破坏位置/未来语义
        key = hash(tuple(seq))
        if key != self._cur_prompt_key:
            self._cur_prompt_key = key
            self._pos_map = {b: i for i, b in enumerate(seq)}
            self._cur_idx = None

    def _will_appear_later(self, bid: int) -> bool:
        if self._cur_idx is None:
            return False
        pos = self._pos_map.get(bid)
        return (pos is not None) and (pos > self._cur_idx)

    # ------------------------ 权重选择（按 type） ------------------------
    def _select_weights_by_type(self, meta: Optional[Dict[str, Any]]) -> Tuple[float, float, float]:
        t = None
        if meta:
            t = (meta.get("type") or meta.get("app") or meta.get("prompt_type"))
            if isinstance(t, str):
                t = t.lower().strip()
        if t in {"chat", "agent", "assistant"}:
            return (1.0, 0.6, 1.8)    # 更重位置（前缀）
        if t in {"search", "rag", "docqa", "qa"}:
            return (1.6, 0.6, 0.9)    # 更重频率
        return (self.W_FREQ, self.W_RECENCY, self.W_POS)

    # ------------------------ keep_score（越大越应保留） ------------------------
    def _pos_score(self, bid: int) -> float:
        pos = self._pos_map.get(bid)
        if pos is None:
            return 0.0
        if pos <= self.POS_HEAD_CUTOFF:
            return 1.0 / (1.0 + pos)
        # 远离头部快速衰减，避免放大尾部位置
        return 1.0 / (1.0 + self.POS_HEAD_CUTOFF + 4.0 * (pos - self.POS_HEAD_CUTOFF))

    def _recency_score(self, bid: int) -> float:
        la = self.last_access.get(bid, 0)
        age = self.step - la
        return 1.0 / (1.0 + max(0, age))

    def _freq_score(self, bid: int) -> float:
        return math.log1p(float(self.freq.get(bid, 0)))  # 平滑，避免碾压

    def _keep_score(self, bid: int, meta: Optional[Dict[str, Any]]) -> float:
        # 低频时偏 LRU（但仍受未来保护）
        if self.freq.get(bid, 0) <= self.FREQ_COLD:
            s = self._recency_score(bid)
            if self._will_appear_later(bid):
                s *= self.FUTURE_MULT
            return s
        wf, wr, wp = self._select_weights_by_type(meta)
        s = wf * self._freq_score(bid) + wr * self._recency_score(bid) + wp * self._pos_score(bid)
        if self._will_appear_later(bid):
            s *= self.FUTURE_MULT
        return s

    # ------------------------ 堆维护工具 ------------------------
    def _pinned_target_count(self) -> int:
        hard_cap = min(int(self.cache_size * self.PREFIX_RESERVE_RATIO), self.PREFIX_MAX_LENGTH)
        ema_cap = int(self.prefix_len_ema + 1e-6)
        return max(0, min(hard_cap, ema_cap))

    def _rebalance_pinned_if_needed(self) -> None:
        new_r = self._pinned_target_count()
        if new_r == self._r_target:
            return
        if new_r < self._r_target:
            k = self._r_target - new_r
            for _ in range(k):
                while self._pin_heap:
                    negid, bid = heapq.heappop(self._pin_heap)
                    if bid in self.cache and bid in self._is_pinned:
                        self._is_pinned.remove(bid)
                        self._push_candidate(bid, self._last_meta)
                        break
            self._r_target = new_r
        else:
            k = new_r - self._r_target
            for _ in range(k):
                nid = self._pop_smallest_unpinned_id()
                if nid is None:
                    break
                self._pin(nid)
            self._r_target = new_r

    def _pin(self, bid: int) -> None:
        if bid in self._is_pinned:
            return
        self._is_pinned.add(bid)
        heapq.heappush(self._pin_heap, (-bid, bid))

    def _push_candidate(self, bid: int, meta: Optional[Dict[str, Any]] = None) -> None:
        if bid in self._is_pinned or bid not in self.cache:
            return
        self._gen += 1
        self._stamp[bid] = self._gen
        s = self._keep_score(bid, meta)
        heapq.heappush(self._cand_heap, (s, self._gen, bid))

    def _pop_smallest_unpinned_id(self) -> Optional[int]:
        while self._all_id_minheap:
            bid = heapq.heappop(self._all_id_minheap)
            if bid in self.cache and bid not in self._is_pinned:
                return bid
        return None

    def _pick_victim(self, meta: Optional[Dict[str, Any]]) -> int:
        """
        受害者选择：
          - 冷启动：LRU + 未来保护
          - 正常：cand_heap 懒失效；若空再从 pin_heap 中选（优先不杀“本轮还会用”的；必要时选“未来距离最远”的）
        """
        if not self.cache:
            raise RuntimeError("cache empty when picking victim")

        # 冷启动：线性 LRU + 未来保护
        if self.step <= self.WARMUP_GLOBAL:
            oldest_id, oldest_age = None, -1
            furthest_id, furthest_d = None, -1
            for b in self.cache:
                pos = self._pos_map.get(b)
                if (self._cur_idx is None) or (pos is None) or (pos <= self._cur_idx):
                    age = self.step - self.last_access.get(b, 0)
                    if age > oldest_age:
                        oldest_age, oldest_id = age, b
                else:
                    d = pos - self._cur_idx
                    if d > furthest_d:
                        furthest_d, furthest_id = d, b
            return oldest_id if oldest_id is not None else furthest_id

        # 正常阶段：候选堆懒失效 + 现时重算
        while self._cand_heap:
            s_old, gen, bid = heapq.heappop(self._cand_heap)
            if (bid not in self.cache) or (bid in self._is_pinned) or (self._stamp.get(bid) != gen):
                continue
            s_now = self._keep_score(bid, meta)
            if abs(s_now - s_old) > self.HEAP_EPS:
                self._gen += 1
                self._stamp[bid] = self._gen
                heapq.heappush(self._cand_heap, (s_now, self._gen, bid))
                continue
            return bid

        # 候选堆空：从前缀里选 —— 仍遵守“未来保护”
        tmp: List[Tuple[int, int]] = []
        best_nonfuture: Tuple[Optional[int], float] = (None, float("inf"))  # (id, keep_score)
        best_future: Tuple[Optional[int], int] = (None, -1)                # (id, distance)
        while self._pin_heap:
            negid, bid = heapq.heappop(self._pin_heap)
            if bid not in self.cache or bid not in self._is_pinned:
                continue
            tmp.append((negid, bid))
            pos = self._pos_map.get(bid)
            if (self._cur_idx is None) or (pos is None) or (pos <= self._cur_idx):
                # 本轮不会再访问：选 keep_score 最小的
                ks = self._keep_score(bid, meta)
                # 这里 _keep_score 已考虑 FUTURE_MULT；但对“不会再访问”的项，等价于原值
                if ks < best_nonfuture[1]:
                    best_nonfuture = (bid, ks)
            else:
                d = pos - self._cur_idx
                if d > best_future[1]:
                    best_future = (bid, d)

        chosen = best_nonfuture[0] if best_nonfuture[0] is not None else best_future[0]
        # 把其余 pin 放回
        for negid, bid in tmp:
            if bid != chosen:
                heapq.heappush(self._pin_heap, (negid, bid))
        if chosen is not None and chosen in self._is_pinned:
            self._is_pinned.remove(chosen)
        return chosen if chosen is not None else max(self.cache)

    def _admit_new_block(self, bid: int, meta: Optional[Dict[str, Any]]) -> None:
        heapq.heappush(self._all_id_minheap, bid)
        self._rebalance_pinned_if_needed()

        if len(self._is_pinned) < self._r_target:
            self._pin(bid)
            return

        if self._pin_heap:
            _, cur_max = self._pin_heap[0]  # 编号最大的前缀
            if bid < cur_max:
                heapq.heappop(self._pin_heap)
                if cur_max in self._is_pinned:
                    self._is_pinned.remove(cur_max)
                    self._push_candidate(cur_max, meta)
                self._pin(bid)
                return

        self._push_candidate(bid, meta)

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

        for d in (self.freq, self.last_access, self._stamp):
            for k in list(d.keys()):
                if k not in keep:
                    d.pop(k, None)

    # ------------------------ 主流程 ------------------------
    def access(self, block_id: int, prompt_blocks: Iterable[int], meta: Dict[str, Any] | None = None) -> bool:
        self.step += 1
        self.access_count += 1
        self._last_meta = meta

        # 建立/更新本轮的位置信息 & 当前索引
        self._ensure_pos_map(prompt_blocks)
        self._cur_idx = self._pos_map.get(block_id, self._cur_idx)

        # 更新前缀 EMA（与你原逻辑一致）
        try:
            self._maybe_update_prefix_ema(prompt_blocks)
        except Exception:
            pass

        # 命中
        if block_id in self.cache:
            self.freq[block_id] += 1
            self.last_access[block_id] = self.step
            if block_id not in self._is_pinned:
                self._push_candidate(block_id, meta)
            if len(self.freq) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size) or \
               len(self.last_access) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size):
                self._maybe_gc_stats()
            return True

        # 未命中：必要时驱逐
        if len(self.cache) >= self.cache_size > 0:
            victim = self._pick_victim(meta)
            if victim in self.cache:
                self.cache.remove(victim)
                self._backend_del(victim)

        # 插入新块（评测：miss 必须 add 并计一次 miss）
        self.cache.add(block_id)
        self.miss += 1
        self._backend_add(block_id)
        self.freq[block_id] += 1
        self.last_access[block_id] = self.step

        self._admit_new_block(block_id, meta)

        if len(self.freq) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size) or \
           len(self.last_access) > self.STATS_GC_MULTIPLIER * max(1, self.cache_size):
            self._maybe_gc_stats()

        return False
