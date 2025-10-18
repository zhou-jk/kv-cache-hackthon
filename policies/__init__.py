from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .base import BaseKVCachePolicy, TraceEvent, compare_policies, load_trace_from_text, run_policy
from .simple import FIFOKVCachePolicy, LRUKVCachePolicy
from .palfu import PrefixAwareKVCachePolicy
from .kvstore import KVCacheStore
from .wa import WorkloadAwareKVCachePolicy
from .weight import WeightedEvictionPolicy
from .kv import KVCachePolicy

__all__ = [
    "BaseKVCachePolicy",
    "TraceEvent",
    "compare_policies",
    "load_trace_from_text",
    "run_policy",
    "create_policy",
]


def create_policy(name: str, args) -> BaseKVCachePolicy:
    kvstore = KVCacheStore(capacity=args.cache_size)
    key = name.strip().lower()
    if key == "fifo":
        return FIFOKVCachePolicy(kvstore)
    if key == "lru":
        return LRUKVCachePolicy(kvstore)
    if key == "kv":
        return KVCachePolicy(kvstore)
    if key in {"pref"}:
        return PrefixAwareKVCachePolicy(
            kvstore,
        )
    if key in {"wa"}:
        return WorkloadAwareKVCachePolicy(
            kvstore,
        )
    if key in {"weight"}:
        return WeightedEvictionPolicy(
            kvstore,
        )
    raise ValueError(f"未知策略名称: {name}")
