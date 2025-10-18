from .base import BaseKVCachePolicy
from .kvstore import KVCacheStore

from .palfu import PrefixAwareKVCachePolicy
from .weight import WeightedEvictionPolicy
from .simple import LRUKVCachePolicy


class KVCachePolicy(BaseKVCachePolicy):
    name = "kv"

    def __init__(self, store: KVCacheStore) -> None:
        super().__init__(store)
        self._policy = self._create_policy()

    def _create_policy(self) -> BaseKVCachePolicy:
        """根据缓存大小选择策略，逻辑简单直接。"""
        size = self.cache_size

        if size < 1000 and PrefixAwareKVCachePolicy is not None:
            return PrefixAwareKVCachePolicy(self.store)

        if size < 20000 and WeightedEvictionPolicy is not None:
            return WeightedEvictionPolicy(self.store)

        if LRUKVCachePolicy is not None:
            return LRUKVCachePolicy(self.store)

        raise RuntimeError("无法加载合适的缓存策略")

    def access(self, block_id, prompt_blocks, type):
        return self._policy.access(block_id, prompt_blocks, type)

    def __getattr__(self, name):
        return getattr(self._policy, name)
