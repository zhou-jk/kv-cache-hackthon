from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score


class AdvancedKVCachePolicy(BaseKVCachePolicy):
    """增强策略：控制前缀保留 + 轻量级预取 + LFU/LRU 混合淘汰。"""

    name = "advanced"

    def __init__(
        self,
        cache_size: int,
        *,
        prefetch_window: int = 4,
        prefetch_depth: int = 0,
        prefix_keep: int = 64,
    ) -> None:
        super().__init__(cache_size)
        self.prefetch_window = max(prefetch_window, 0)
        self.prefetch_depth = max(prefetch_depth, 0)
        self.prefix_keep = max(prefix_keep, 0)
        self.access_count = 0
        self.miss = 0

    def init(self) -> None:
        self.step = 0
        self.entries: Dict[int, BlockEntry] = {}
        self.freq_counter: Dict[int, int] = {}
        self.access_count = 0
        self.miss = 0

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, int]:
        del meta  # 当前策略未使用额外元数据
        self.access_count += 1
        self.step += 1

        prompt_list = list(prompt_blocks)
        idx = self._locate_block(block_id, prompt_list)
        score = calc_prefix_score(block_id, prompt_list)

        freq = self.freq_counter.get(block_id, 0) + 1
        self.freq_counter[block_id] = freq

        added = 0

        entry = self.entries.get(block_id)
        if entry is not None:
            entry.last_access = self.step
            entry.freq = freq
            entry.prefix_score = max(entry.prefix_score, score)
            if idx >= 0 and idx < self.prefix_keep:
                entry.pinned = True
            if idx == 0:
                added += self._ensure_prompt_prefix(prompt_list)
            added += self._prefetch_forward(prompt_list, idx)
            return True, added

        pinned = idx >= 0 and idx < self.prefix_keep
        if self._admit(block_id, score, freq, pinned):
            added += 1
            self.miss += 1

        if idx == 0:
            added += self._ensure_prompt_prefix(prompt_list)

        added += self._prefetch_forward(prompt_list, idx)
        return False, added

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    def _prefetch_forward(self, prompt_list: List[int], index: int) -> int:
        if index < 0 or self.prefetch_window <= 0:
            return 0

        added = 0
        max_offset = min(
            self.prefetch_window,
            len(prompt_list) - index - 1,
            self.prefix_keep - index - 1,
        )
        for offset in range(1, max_offset + 1):
            candidate = prompt_list[index + offset]
            if candidate in self.entries:
                continue
            score = calc_prefix_score(candidate, prompt_list)
            freq = self.freq_counter.get(candidate, 0)
            pinned = (index + offset) < self.prefix_keep
            if self._admit(candidate, score, freq, pinned):
                added += 1
                self.miss += 1
        return added

    def _ensure_prompt_prefix(self, prompt_list: List[int]) -> int:
        limit = min(len(prompt_list), self.prefix_keep)
        added = 0
        for i in range(limit):
            block_id = prompt_list[i]
            score = calc_prefix_score(block_id, prompt_list)
            freq = self.freq_counter.get(block_id, 0)
            if self._admit(block_id, score, freq, pinned=True):
                added += 1
                self.miss += 1
        return added

    def _admit(self, block_id: int, score: float, freq: int, pinned: bool) -> bool:
        if self.cache_size <= 0:
            return False

        entry = self.entries.get(block_id)
        if entry is not None:
            entry.last_access = self.step
            entry.freq = freq
            entry.prefix_score = max(entry.prefix_score, score)
            if pinned:
                entry.pinned = True
            return False

        if len(self.entries) >= self.cache_size:
            victim = self._select_victim(prefer_eviction_of_pinned=pinned)
            if victim is None:
                return False
            self.entries.pop(victim.block_id, None)

        self.entries[block_id] = BlockEntry(
            block_id=block_id,
            last_access=self.step,
            freq=freq,
            prefix_score=score,
            pinned=pinned,
        )
        return True

    def _select_victim(self, prefer_eviction_of_pinned: bool) -> Optional[BlockEntry]:
        candidates = [
            entry for entry in self.entries.values() if not entry.pinned
        ]
        if not candidates and prefer_eviction_of_pinned:
            candidates = list(self.entries.values())
        if not candidates:
            return None
        return min(candidates, key=lambda e: (e.freq, e.prefix_score, e.last_access))

    @staticmethod
    def _locate_block(block_id: int, prompt_list: List[int]) -> int:
        for idx in range(len(prompt_list) - 1, -1, -1):
            if prompt_list[idx] == block_id:
                return idx
        return -1
