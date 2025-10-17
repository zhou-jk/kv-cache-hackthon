"""仅使用 GPU 层的 KV Cache 策略实现。

核心思路：
1. 以 block 粒度维护缓存，命中时更新最近访问步数、访问频率、前缀得分；
2. 淘汰时综合考虑“距离上次访问的间隔”“使用频率”“在 prompt 中的位置”“是否需要 pin”四类因素；
3. 结合题面中的“公共前缀”特性，在每次访问时对当前 block 前面的若干块进行前缀预取；
4. 支持命令行入口，可读取 .jsonl trace（字段默认为 hash_ids）进行评估。

整个实现只包含单层缓存（模拟 GPU KV cache），方便在没有 CPU 存储层的环境下直接使用。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TraceEvent = Tuple[List[int], int, Dict[str, Any] | None]


# ---------------------------------------------------------------------------
# 数据结构与缓存层
# ---------------------------------------------------------------------------


@dataclass
class BlockEntry:
    """单个 block 的缓存元数据。"""

    block_id: int
    last_access: int
    freq: int
    prefix_score: float
    pinned: bool = False


class CacheTier:
    """单层缓存，负责插入/淘汰/命中判断。"""

    def __init__(
        self,
        capacity_blocks: int,
        *,
        recency_weight: float = 1.2,
        freq_weight: float = 1.6,
        prefix_weight: float = 2.8,
        pin_weight: float = 40.0,
    ) -> None:
        self.capacity = max(capacity_blocks, 1)
        self.recency_weight = recency_weight
        self.freq_weight = freq_weight
        self.prefix_weight = prefix_weight
        self.pin_weight = pin_weight
        self.entries: Dict[int, BlockEntry] = {}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.entries

    def get(self, block_id: int) -> Optional[BlockEntry]:
        return self.entries.get(block_id)

    def add(
        self,
        entry: BlockEntry,
        current_step: int,
    ) -> None:
        if entry.block_id in self.entries:
            return
        self._ensure_capacity(current_step)
        self.entries[entry.block_id] = entry

    def _ensure_capacity(self, current_step: int) -> None:
        while len(self.entries) >= self.capacity:
            victim = self._select_victim(current_step)
            if victim is None:
                break
            self.entries.pop(victim.block_id, None)

    def _select_victim(self, current_step: int) -> Optional[BlockEntry]:
        if not self.entries:
            return None
        return max(
            self.entries.values(),
            key=lambda entry: self._eviction_penalty(entry, current_step),
        )

    def _eviction_penalty(self, entry: BlockEntry, current_step: int) -> float:
        recency_cost = self.recency_weight * (current_step - entry.last_access)
        freq_bonus = self.freq_weight * entry.freq
        prefix_bonus = self.prefix_weight * entry.prefix_score
        pin_bonus = self.pin_weight if entry.pinned else 0.0
        return recency_cost - freq_bonus - prefix_bonus - pin_bonus

    @property
    def used(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# 策略主类
# ---------------------------------------------------------------------------


class SingleTierKVCachePolicy:
    """使用单层缓存的 KV 策略。"""

    def __init__(
        self,
        cache_size: int,
        *,
        prefetch_window: int = 32,
        prefetch_depth: int = 3,
    ) -> None:
        self.cache_size = max(cache_size, 1)
        self.prefetch_window = max(prefetch_window, 1)
        self.prefetch_depth = max(prefetch_depth, 0)
        self.tier = CacheTier(self.cache_size)
        self.step = 0

    # ------------------------------ 公共接口 ----------------------------- #

    def init(self) -> None:
        self.tier = CacheTier(
            self.cache_size,
            recency_weight=self.tier.recency_weight,
            freq_weight=self.tier.freq_weight,
            prefix_weight=self.tier.prefix_weight,
            pin_weight=self.tier.pin_weight,
        )
        self.step = 0

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        self.step += 1
        prompt_list = list(prompt_blocks)
        prefix_score = self._calc_prefix_score(block_id, prompt_list)

        entry = self.tier.get(block_id)
        if entry is not None:
            self._touch(entry, prefix_score)
            self._prefetch(prompt_list, block_id, meta)
            self._maybe_pin(entry, meta)
            return True

        entry = BlockEntry(
            block_id=block_id,
            last_access=self.step,
            freq=1,
            prefix_score=prefix_score,
        )
        self.tier.add(entry, self.step)
        self._prefetch(prompt_list, block_id, meta)
        self._maybe_pin(entry, meta)
        return False

    # ------------------------------ 工具逻辑 ----------------------------- #

    def _touch(self, entry: BlockEntry, prefix_score: float) -> None:
        entry.last_access = self.step
        entry.freq += 1
        entry.prefix_score = max(entry.prefix_score, prefix_score)

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
        if meta:
            inp = meta.get("input_length") or meta.get("prompt_length")
            if isinstance(inp, int) and inp > self.prefetch_window * 3:
                depth = min(depth + 1, depth + 2)

        for offset in range(1, depth + 1):
            target_idx = index - offset
            if target_idx < 0:
                break
            candidate_id = prompt_list[target_idx]
            if candidate_id in self.tier:
                continue
            score = self._calc_prefix_score(candidate_id, prompt_list)
            entry = BlockEntry(
                block_id=candidate_id,
                last_access=self.step,
                freq=1,
                prefix_score=score,
            )
            self.tier.add(entry, self.step)

    def _maybe_pin(self, entry: BlockEntry, meta: Dict[str, Any] | None) -> None:
        if meta is None:
            return
        if meta.get("round", 0) == 0 or meta.get("turn", 0) == 0:
            entry.pinned = True
            return
        session_type = meta.get("session_type") or meta.get("app_type")
        if session_type in {"system", "instruction", "retrieval"}:
            entry.pinned = True
            return
        input_len = meta.get("input_length") or meta.get("prompt_length")
        if isinstance(input_len, int) and input_len >= self.prefetch_window * 5:
            entry.pinned = True

    @staticmethod
    def _calc_prefix_score(block_id: int, prompt_list: List[int]) -> float:
        length = len(prompt_list)
        if length == 0:
            return 0.0
        try:
            index = prompt_list.index(block_id)
        except ValueError:
            index = length - 1
        return (length - index) / length


# ---------------------------------------------------------------------------
# Trace 装载与评估
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


def simulate(
    policy: SingleTierKVCachePolicy,
    events: Iterable[TraceEvent],
) -> Tuple[int, int]:
    policy.init()
    hits = 0
    total = 0
    for prompt_blocks, block_id, meta in events:
        total += 1
        if policy.access(block_id, prompt_blocks, meta):
            hits += 1
    return hits, total


def build_sample_trace() -> List[TraceEvent]:
    prompts = [
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 6],
    ]
    events: List[TraceEvent] = []
    for prompt in prompts:
        meta = {"input_length": len(prompt), "round": 0}
        for block in prompt:
            events.append((prompt, block, meta))
    return events


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="单层 KV Cache 策略评估工具")
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
    args = parser.parse_args()

    if args.trace:
        events = load_trace_from_jsonl(args.trace, block_field=args.block_field)
    else:
        events = build_sample_trace()

    policy = SingleTierKVCachePolicy(
        cache_size=args.cache_size,
        prefetch_window=args.prefetch_window,
        prefetch_depth=args.prefetch_depth,
    )
    hits, total = simulate(policy, events)
    hit_rate = hits / total if total else 0.0

    print("=== KV Cache 策略评估结果 ===")
    print(f"- 总访问: {total}")
    print(f"- 命中数: {hits}")
    print(f"- 命中率: {hit_rate:.2%}")
    print(f"- GPU 占用: {policy.tier.used}/{policy.cache_size}")


if __name__ == "__main__":
    main()
