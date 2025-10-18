from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
from .kvstore import KVCacheStore

TraceEvent = Tuple[List[int], int, Dict[str, Any] | None]

@dataclass
class BlockEntry:
    block_id: int
    last_access: int
    freq: int
    prefix_score: float
    pinned: bool = False


class BaseKVCachePolicy:
    name: str = "base"
    miss: int = 0
    access_count: int = 0

    def __init__(self, store: KVCacheStore) -> None:
        self.store = store
        self.cache_size = max(store.capacity, 1)

    def access(  # pragma: no cover
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        type: int,
    ) -> bool:
        raise NotImplementedError


def calc_prefix_score(block_id: int, prompt_list: List[int]) -> float:
    length = len(prompt_list)
    if length == 0:
        return 0.0
    try:
        index = prompt_list.index(block_id)
    except ValueError:
        index = length - 1
    return (length - index) / length


def load_trace_from_text(
    path: Path,
) -> Tuple[int, List[TraceEvent]]:
    """同时支持 JSONL 与简单文本格式。

    文本格式示例：
        3
        {1,2,3,4} 1
        {1,2,3,4,5} 1
        {1,2,6} 2
    """

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [line.strip() for line in raw_lines if line.strip()]
    if not stripped:
        return []

    lines = stripped
    cache_size = 0
    if lines and lines[0].isdigit():
        cache_size = int(lines[0])
        lines = lines[1:]
    events: List[TraceEvent] = []

    for line_no, line in enumerate(lines, 1):
        if "{" not in line or "}" not in line:
            raise ValueError(f"无法解析第 {line_no} 行：{line}")
        lbr = line.index("{")
        rbr = line.index("}", lbr)
        blocks_part = line[lbr + 1 : rbr]
        block_items = [item.strip() for item in blocks_part.split(",") if item.strip()]
        try:
            prompt_blocks = [int(item) for item in block_items]
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"第 {line_no} 行无法解析 block 列表：{line}") from exc

        tail = line[rbr + 1 :].strip()
        type = 0
        if tail:
            token = tail.split()[0]
            type = int(token)

        for block_id in prompt_blocks:
            events.append((prompt_blocks, block_id, type))

    return cache_size, events


def run_policy(policy: BaseKVCachePolicy, events: Iterable[TraceEvent]) -> Tuple[int, int]:
    hits = 0
    total = 0
    for prompt_blocks, block_id, meta in events:
        total += 1
        if policy.access(block_id, prompt_blocks, meta):
            hits += 1
        if not policy.store.contains(block_id):
            raise RuntimeError(f"策略 {policy.name} 未能正确缓存 block {block_id}")
    return policy.store.miss, total


def compare_policies(policies: Sequence[BaseKVCachePolicy], events: Iterable[TraceEvent]) -> None:
    print("=== 策略对比 ===")
    header = "{:<12} {:>12} {:>12} {:>12}".format("Policy", "Total", "Hits", "HitRate")
    print(header)
    print("-" * len(header))
    for policy in policies:
        miss, total = run_policy(policy, events)
        hit = total - miss
        hit_rate = hit / total if total else 0.0
        print(
            "{:<12} {:>12} {:>12} {:>11.2%}".format(
                policy.name, total, hit, hit_rate
            )
        )
    print("=" * len(header))
