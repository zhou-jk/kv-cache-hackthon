"""单节点 GPU KV 缓存策略对比框架。

提供多种策略实现（FIFO、LRU、前缀感知、增强版）并支持直接对比。
所有策略遵循相同的 `init` / `access` 接口，便于在相同 trace 上评估命中率。
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TraceEvent = Tuple[List[int], int, Dict[str, Any] | None]


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def calc_prefix_score(block_id: int, prompt_list: List[int]) -> float:
    """越靠前的 block 得分越高，用于鼓励保留公共前缀。"""

    length = len(prompt_list)
    if length == 0:
        return 0.0
    try:
        index = prompt_list.index(block_id)
    except ValueError:
        index = length - 1
    return (length - index) / length


@dataclass
class BlockEntry:
    block_id: int
    last_access: int
    freq: int
    prefix_score: float
    pinned: bool = False
    hot_score: float = 1.0
    expires_at: int = 0
    compressed: bool = False


class BaseKVCachePolicy:
    """策略基类，所有策略需实现 init / access。"""

    name: str = "base"

    def __init__(self, cache_size: int) -> None:
        self.cache_size = max(cache_size, 1)

    def init(self) -> None:  # pragma: no cover - 子类覆写
        raise NotImplementedError

    def access(  # pragma: no cover - 子类覆写
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 简单策略：FIFO / LRU / 前缀感知
# ---------------------------------------------------------------------------


class FIFOKVCachePolicy(BaseKVCachePolicy):
    name = "fifo"

    def init(self) -> None:
        self.queue: List[int] = []
        self.cache: set[int] = set()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        del prompt_blocks, meta
        if block_id in self.cache:
            return True
        if len(self.cache) >= self.cache_size:
            victim = self.queue.pop(0)
            self.cache.remove(victim)
        self.cache.add(block_id)
        self.queue.append(block_id)
        return False


class LRUKVCachePolicy(BaseKVCachePolicy):
    name = "lru"

    def init(self) -> None:
        self.table: OrderedDict[int, None] = OrderedDict()

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        del prompt_blocks, meta
        if block_id in self.table:
            self.table.move_to_end(block_id)
            return True
        if len(self.table) >= self.cache_size:
            self.table.popitem(last=False)
        self.table[block_id] = None
        return False


class PrefixAwareKVCachePolicy(BaseKVCachePolicy):
    """简单的前缀感知策略：结合频次/最近访问/前缀权重。"""

    name = "prefix"

    def __init__(
        self,
        cache_size: int,
        *,
        prefetch_depth: int = 0,
    ) -> None:
        super().__init__(cache_size)
        self.prefetch_depth = max(prefetch_depth, 0)

    def init(self) -> None:
        self.step = 0
        self.entries: Dict[int, BlockEntry] = {}

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        del meta
        self.step += 1
        prompt_list = list(prompt_blocks)
        score = calc_prefix_score(block_id, prompt_list)
        entry = self.entries.get(block_id)
        if entry is not None:
            entry.last_access = self.step
            entry.freq += 1
            entry.prefix_score = max(entry.prefix_score, score)
            return True

        if len(self.entries) >= self.cache_size:
            victim_id = max(
                self.entries,
                key=lambda bid: self._eviction_penalty(self.entries[bid]),
            )
            self.entries.pop(victim_id, None)

        self.entries[block_id] = BlockEntry(
            block_id=block_id,
            last_access=self.step,
            freq=1,
            prefix_score=score,
        )

        if self.prefetch_depth > 0:
            self._prefetch(prompt_list, block_id)
        return False

    def _eviction_penalty(self, entry: BlockEntry) -> float:
        return (
            (self.step - entry.last_access)
            - 1.2 * entry.freq
            - 3.0 * entry.prefix_score
        )

    def _prefetch(self, prompt_list: List[int], block_id: int) -> None:
        try:
            index = len(prompt_list) - 1 - prompt_list[::-1].index(block_id)
        except ValueError:
            return
        for offset in range(1, self.prefetch_depth + 1):
            target = index - offset
            if target < 0:
                break
            candidate = prompt_list[target]
            if candidate in self.entries:
                continue
            score = calc_prefix_score(candidate, prompt_list)
            if len(self.entries) >= self.cache_size:
                victim_id = max(
                    self.entries,
                    key=lambda bid: self._eviction_penalty(self.entries[bid]),
                )
                if self._eviction_penalty(self.entries[victim_id]) <= 0:
                    break
                self.entries.pop(victim_id, None)
            self.entries[candidate] = BlockEntry(
                block_id=candidate,
                last_access=self.step,
                freq=1,
                prefix_score=score,
            )


# ---------------------------------------------------------------------------
# 增强策略（带 TTL / 热度 / 前缀索引 / 压缩仓库）
# ---------------------------------------------------------------------------


class AdvancedKVCachePolicy(BaseKVCachePolicy):
    name = "advanced"

    def __init__(
        self,
        cache_size: int,
        *,
        prefetch_window: int = 32,
        prefetch_depth: int = 3,
        prefix_keep: int = 64,
    ) -> None:
        super().__init__(cache_size)
        self.prefetch_window = max(prefetch_window, 1)
        self.prefetch_depth = max(prefetch_depth, 0)
        self.prefix_keep = max(prefix_keep, 0)

    def init(self) -> None:
        self.step = 0
        self.entries: Dict[int, BlockEntry] = {}
        self.freq_counter: Dict[int, int] = {}

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        self.step += 1
        prompt_list = list(prompt_blocks)
        score = calc_prefix_score(block_id, prompt_list)
        idx = self._locate_block(block_id, prompt_list)

        freq = self.freq_counter.get(block_id, 0) + 1
        self.freq_counter[block_id] = freq

        entry = self.entries.get(block_id)
        if entry is not None:
            entry.last_access = self.step
            entry.freq = freq
            entry.prefix_score = max(entry.prefix_score, score)
            if idx >= 0 and idx < self.prefix_keep:
                entry.pinned = True
            if idx == 0:
                self._ensure_prompt_prefix(prompt_list)
            return True

        if len(self.entries) >= self.cache_size:
            victim = self._select_victim()
            if victim is None:
                return False
            if freq <= victim.freq and score <= victim.prefix_score:
                return False
            self.entries.pop(victim.block_id, None)

        self.entries[block_id] = BlockEntry(
            block_id=block_id,
            last_access=self.step,
            freq=freq,
            prefix_score=score,
            pinned=idx >= 0 and idx < self.prefix_keep,
        )

        if idx == 0:
            self._ensure_prompt_prefix(prompt_list)

        self._prefetch(prompt_list, block_id, meta)
        return False

    # ------------------------------ 内部逻辑 ----------------------------- #

    def _prefetch(
        self,
        prompt_list: List[int],
        block_id: int,
        meta: Dict[str, Any] | None,
    ) -> None:
        try:
            index = len(prompt_list) - 1 - prompt_list[::-1].index(block_id)
        except ValueError:
            index = len(prompt_list) - 1

        depth = self.prefetch_depth
        for offset in range(1, depth + 1):
            target_idx = index - offset
            if target_idx < 0:
                break
            candidate_id = prompt_list[target_idx]
            if candidate_id in self.entries:
                continue
            score = calc_prefix_score(candidate_id, prompt_list)
            freq = self.freq_counter.get(candidate_id, 0)
            if len(self.entries) >= self.cache_size:
                victim = self._select_victim()
                if victim is None:
                    break
                if freq <= victim.freq and score <= victim.prefix_score:
                    continue
                self.entries.pop(victim.block_id, None)
            self.entries[candidate_id] = BlockEntry(
                block_id=candidate_id,
                last_access=self.step,
                freq=freq,
                prefix_score=score,
                pinned=target_idx < self.prefix_keep,
            )

    def _locate_block(self, block_id: int, prompt_list: List[int]) -> int:
        for idx in range(len(prompt_list) - 1, -1, -1):
            if prompt_list[idx] == block_id:
                return idx
        return -1

    def _select_victim(self) -> Optional[BlockEntry]:
        candidates = [entry for entry in self.entries.values() if not entry.pinned]
        if not candidates:
            candidates = list(self.entries.values())
            if not candidates:
                return None
        return min(candidates, key=lambda e: (e.freq, e.prefix_score, e.last_access))

    def _ensure_prompt_prefix(self, prompt_list: List[int]) -> None:
        limit = min(len(prompt_list), self.prefetch_window)
        for i in range(limit):
            block_id = prompt_list[i]
            freq = self.freq_counter.get(block_id, 0)
            existing = self.entries.get(block_id)
            score = calc_prefix_score(block_id, prompt_list)
            if existing:
                existing.pinned = True
                existing.prefix_score = max(existing.prefix_score, score)
                continue
            if len(self.entries) >= self.cache_size:
                victim = self._select_victim()
                if victim is None:
                    break
                if freq <= victim.freq and score <= victim.prefix_score:
                    continue
                self.entries.pop(victim.block_id, None)
            self.entries[block_id] = BlockEntry(
                block_id=block_id,
                last_access=self.step,
                freq=freq,
                prefix_score=score,
                pinned=i < self.prefix_keep,
            )


# ---------------------------------------------------------------------------
# Trace 读取与评估
# ---------------------------------------------------------------------------


def load_trace_from_jsonl(
    path: Path,
    *,
    block_field: str = "hash_ids",
) -> List[TraceEvent]:
    events: List[TraceEvent] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, 1):
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            blocks = record.get(block_field)
            if not isinstance(blocks, list):
                raise ValueError(f"第 {line_no} 行缺少字段 {block_field}")
            prompt_blocks = [int(b) for b in blocks]
            meta = {k: v for k, v in record.items() if k != block_field} or None
            for block_id in prompt_blocks:
                events.append((prompt_blocks, block_id, meta))
    return events


def run_policy(policy: BaseKVCachePolicy, events: Iterable[TraceEvent]) -> Tuple[int, int]:
    policy.init()
    hits = 0
    total = 0
    for prompt_blocks, block_id, meta in events:
        total += 1
        if policy.access(block_id, prompt_blocks, meta):
            hits += 1
    return hits, total


def compare_policies(policies: Sequence[BaseKVCachePolicy], events: Iterable[TraceEvent]) -> None:
    print("=== 策略对比 ===")
    header = "{:<12} {:>12} {:>12} {:>12}".format("Policy", "Total", "Hits", "HitRate")
    print(header)
    print("-" * len(header))
    for policy in policies:
        hits, total = run_policy(policy, events)
        hit_rate = hits / total if total else 0.0
        print(
            "{:<12} {:>12} {:>12} {:>11.2%}".format(
                policy.name, total, hits, hit_rate
            )
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def create_policy(name: str, args: argparse.Namespace) -> BaseKVCachePolicy:
    key = name.strip().lower()
    if key in {"fifo"}:
        return FIFOKVCachePolicy(args.cache_size)
    if key in {"lru"}:
        return LRUKVCachePolicy(args.cache_size)
    if key in {"prefix", "pref"}:
        return PrefixAwareKVCachePolicy(
            args.cache_size,
            prefetch_depth=args.prefetch_depth,
        )
    if key in {"advanced", "adv"}:
        return AdvancedKVCachePolicy(
            args.cache_size,
            prefetch_window=args.prefetch_window,
            prefetch_depth=args.prefetch_depth,
            prefix_keep=args.prefix_keep,
        )
    raise ValueError(f"未知策略名称: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache 策略评估工具")
    parser.add_argument("--trace", type=Path, help="JSONL trace 文件路径")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=256,
        help="GPU 层可存储的 block 数",
    )
    parser.add_argument(
        "--prefetch-window",
        type=int,
        default=32,
        help="用于调节预取/固定阈值的窗口大小",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=3,
        help="向前预取的最大层数",
    )
    parser.add_argument(
        "--block-field",
        default="hash_ids",
        help="JSONL 中存放 block 列表的字段名",
    )
    parser.add_argument(
        "--prefix-keep",
        type=int,
        default=64,
        help="增强策略：优先保留每个 prompt 的前多少个 block",
    )
    parser.add_argument(
        "--policies",
        default="advanced",
        help="要运行的策略列表，逗号分隔（如：fifo,lru,prefix,advanced）",
    )
    args = parser.parse_args()

    if args.trace:
        events = load_trace_from_jsonl(args.trace, block_field=args.block_field)
    else:
        sample = [
            ([1, 2, 3, 4], {"input_length": 4, "round": 0, "type": 1}),
            ([1, 2, 3, 4, 5], {"input_length": 5, "round": 1, "type": 1}),
            ([1, 2, 6], {"input_length": 3, "round": 1, "type": 2}),
        ]
        events = []
        for blocks, meta in sample:
            for block in blocks:
                events.append((blocks, block, meta))

    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    if not policy_names:
        raise ValueError("必须至少指定一种策略")

    policies = [create_policy(name, args) for name in policy_names]

    if len(policies) == 1:
        policy = policies[0]
        hits, total = run_policy(policy, events)
        hit_rate = hits / total if total else 0.0

        print("=== KV Cache 策略评估结果 ===")
        print(f"策略: {policy.name}")
        print(f"- 总访问: {total}")
        print(f"- 命中数: {hits}")
        print(f"- 命中率: {hit_rate:.2%}")
    else:
        compare_policies(policies, events)


if __name__ == "__main__":
    main()
