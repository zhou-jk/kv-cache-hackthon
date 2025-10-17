from collections import deque
from kvstore import KVCacheStore


class KVCachePolicy:
    """FIFO 策略（调用 store 的 add/delete 控制容量）。
    """

    def __init__(self, store: KVCacheStore):
        self.store = store
        self.queue = deque()  # FIFO 顺序，仅作为策略内部的淘汰依据

    def access(self, key: int, request_prefix_hash_ids, request_type) -> bool:
        if self.store.contains(key):
            return True
        
        if key not in self.store.store:
            # if full, evict one
            if self.store.size() >= self.store.capacity:
                if not self.queue:
                    raise RuntimeError("KVCachePolicy internal error: queue empty but store full")
                oldest = self.queue.popleft()
                self.store.delete(oldest)
            # add new one
            self.queue.append(key)
            self.store.add(key)
        return False

    def current_keys(self):
        return list(self.queue)
