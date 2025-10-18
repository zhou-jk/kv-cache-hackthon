from __future__ import annotations

from typing import Dict, Iterable, List
from collections import OrderedDict, defaultdict

from .base import BaseKVCachePolicy
from sortedcontainers import SortedList
from heapdict import heapdict
import math
from .kvstore import KVCacheStore

class FIFOKVCachePolicy(BaseKVCachePolicy):
    name = "fifo"
    miss: int = 0
    access_count: int = 0

    def __init__(self, store: KVCacheStore) -> None:
        self.queue: List[int] = []
        self.store = store
        self.cache_size = max(store.capacity, 1)

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        type: int,
    ) -> bool:
        del prompt_blocks, type
        self.access_count += 1
        if self.store.contains(block_id):
            return True
        if self.store.size() >= self.cache_size:
            victim = self.queue.pop(0)
            self.store.delete(victim)
        self.store.add(block_id)
        self.queue.append(block_id)
        self.miss += 1
        return False


class LRUKVCachePolicy(BaseKVCachePolicy):
    name = "lru"
    miss: int = 0
    access_count: int = 0

    def __init__(self, store: KVCacheStore) -> None:
        self.store = store
        self.cache_size = max(store.capacity, 1)
        self.table: OrderedDict[int, None] = OrderedDict()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        type: int,
    ) -> bool:
        self.access_count += 1
        del prompt_blocks, type
        if block_id in self.table:
            self.table.move_to_end(block_id)
            return True
        if len(self.table) >= self.cache_size:
            self.store.delete(next(iter(self.table)))
            self.table.popitem(last=False)
        self.table[block_id] = None
        self.miss += 1
        self.store.add(block_id)
        return False
