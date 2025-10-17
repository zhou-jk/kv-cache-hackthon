from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from .base import BaseKVCachePolicy, TraceEvent, compare_policies, load_trace_from_jsonl, run_policy
from .simple import FIFOKVCachePolicy, LRUKVCachePolicy, PrefixAwareKVCachePolicy
from .advanced import AdvancedKVCachePolicy
from .workload import WorkloadAwareKVCachePolicy

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
    if key in {"workload"}:
        return WorkloadAwareKVCachePolicy(
            args.cache_size,
        )
    if key in {"advanced", "adv"}:
        return AdvancedKVCachePolicy(
            args.cache_size,
            prefetch_window=args.prefetch_window,
            prefetch_depth=args.prefetch_depth,
            prefix_keep=args.prefix_keep,
        )
    raise ValueError(f"未知策略名称: {name}")
