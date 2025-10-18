from __future__ import annotations
from dataclasses import dataclass
import heapq
from collections import OrderedDict
from typing import Dict, Iterable, Optional
from .base import BaseKVCachePolicy
from .kvstore import KVCacheStore

@dataclass
class BlockEntry:
    block_id: int
    insert_time: int
    last_access: int
    freq: int = 1

class WeightedEvictionPolicy(BaseKVCachePolicy):
    """
    LRU + 打分 + 懒失效堆（不再依赖 SCAN_LIMIT）
    - 命中：标准 LRU（move_to_end）+ 重新压分数快照
    - 驱逐：从“大根堆”弹出当前分数最大的有效项作为受害者
    - 复杂度：命中/插入 O(log N)，驱逐 O(log N)
    """
    name = "weight"

    # ---- 权重（和你的一样；w_age 可以为 0）----
    w_age: float = 0.0
    w_recency: float = 0.8
    w_freq: float = 0.1
    w_blockid: float = 0.1

    miss: int = 0
    access_count: int = 0

    def __init__(self, store: KVCacheStore) -> None:
        self.store = store
        self.cache_size = max(store.capacity, 1)
        self.table: OrderedDict[int, BlockEntry] = OrderedDict()
        self.time_step: int = 0
        self._heap: list[tuple[float, int, int]] = []
        self._stamp: Dict[int, int] = {}
        self._gen: int = 0

    def _score_key(self, e: BlockEntry) -> tuple:
        age = self.time_step - e.insert_time
        recency = self.time_step - e.last_access + 1
        freq = e.freq
        bid = max(1, e.block_id)

        norm_age = min(age / 1.0, 1.0)
        norm_recency = 1.0 / float(recency)
        norm_freq = min(freq / 10.0, 1.0)
        norm_blockid = 1.0 / float(bid)

        score = (
            self.w_age      * norm_age +
            self.w_recency  * (1.0 - norm_recency) +
            self.w_freq     * (1.0 - norm_freq) +
            self.w_blockid  * (1.0 - norm_blockid)
        )
        return (score, self.time_step - e.last_access, e.block_id)

    def _heap_push(self, bid: int) -> None:
        e = self.table.get(bid)
        if e is None:
            return
        self._gen += 1
        self._stamp[bid] = self._gen
        score, oldness, _ = self._score_key(e)
        heapq.heappush(self._heap, (-score, -oldness, -e.block_id, self._gen, bid))

    def _pick_victim(self) -> Optional[int]:
        while self._heap:
            neg_score, neg_old, neg_bid, gen, bid = heapq.heappop(self._heap)
            if (bid not in self.table) or (self._stamp.get(bid) != gen):
                continue
            return bid
        return next(iter(self.table)) if self.table else None

    @staticmethod
    def _try_call(fn: str, bid: int) -> None:
        f = globals().get(fn)
        if callable(f):
            try: f(bid)
            except Exception: pass

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        type: int,
    ) -> bool:
        del prompt_blocks, type
        self.access_count += 1
        self.time_step += 1

        if block_id in self.table:
            e = self.table[block_id]
            e.last_access = self.time_step
            e.freq += 1
            self.table.move_to_end(block_id, last=True)
            self._heap_push(block_id)
            return True

        if self.cache_size > 0 and len(self.table) >= self.cache_size:
            victim = self._pick_victim()
            if victim is not None and victim in self.table:
                self.table.pop(victim, None)
                self.store.delete(victim)

        self.table[block_id] = BlockEntry(
            block_id=block_id,
            insert_time=self.time_step,
            last_access=self.time_step,
            freq=1,
        )
        self._heap_push(block_id)
        self.miss += 1
        self.store.add(block_id)
        return False
