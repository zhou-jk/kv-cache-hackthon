
class KVCacheStore:
    def __init__(self, capacity):
        self.capacity = capacity
        self.store = {}

    def add(self, prefix_hash_id):
        if prefix_hash_id in self.store:
            return
        if len(self.store) >= self.capacity:
            raise RuntimeError("KVCacheStore is at capacity; cannot add")
        self.store[prefix_hash_id] = True
    
    def delete(self, prefix_hash_id):
        if prefix_hash_id in self.store:
            del self.store[prefix_hash_id]

    def contains(self, prefix_hash_id):
        return prefix_hash_id in self.store

    def size(self):
        return len(self.store)
