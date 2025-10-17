from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .base import BaseKVCachePolicy, TraceEvent, compare_policies, load_trace_from_jsonl, run_policy
from .simple import FIFOKVCachePolicy, LRUKVCachePolicy, PrefixAwareKVCachePolicy
from .workload import WorkloadAwareKVCachePolicy
from .zjk import PagedEvictionKVCachePolicy
from .s3fifo import S3FIFOKVCachePolicy
from .complex import LRUKKVCachePolicy

__all__ = [
    "BaseKVCachePolicy",
    "TraceEvent",
    "compare_policies",
    "load_trace_from_jsonl",
    "run_policy",
    "create_policy",
]


def create_policy(name: str, args) -> BaseKVCachePolicy:
    key = name.strip().lower()
    if key == "fifo":
        return FIFOKVCachePolicy(args.cache_size)
    if key == "lru":
        return LRUKVCachePolicy(args.cache_size)
    if key in {"prefix", "pref"}:
        return PrefixAwareKVCachePolicy(
            args.cache_size,
        )
    if key in {"wa"}:
        return WorkloadAwareKVCachePolicy(
            args.cache_size,
        )
    if key in {"zjk"}:
        return PagedEvictionKVCachePolicy(
            args.cache_size,
        )
    if key in {"complex"}:
        return LRUKKVCachePolicy(
            args.cache_size,
        )
    if key in {"s3"}:
        return S3FIFOKVCachePolicy(
            args.cache_size,
        )
    raise ValueError(f"未知策略名称: {name}")
