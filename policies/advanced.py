from __future__ import annotations

from typing import Dict, Iterable, List

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score


class AdvancedKVCachePolicy(BaseKVCachePolicy):
    """面向 prefix-heavy trace 的 LFU + 前缀复用策略。"""

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
        meta: Dict | None = None,
    ) -> bool:
        del meta
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

        self._prefetch(prompt_list, block_id)
        return False

    # ------------------------------ 内部逻辑 ----------------------------- #

    def _prefetch(self, prompt_list: List[int], block_id: int) -> None:
        try:
            index = len(prompt_list) - 1 - prompt_list[::-1].index(block_id)
        except ValueError:
            index = len(prompt_list) - 1

        for offset in range(1, self.prefetch_depth + 1):
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

    def _select_victim(self) -> BlockEntry | None:
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
            score = calc_prefix_score(block_id, prompt_list)
            entry = self.entries.get(block_id)
            if entry:
                entry.pinned = True
                entry.prefix_score = max(entry.prefix_score, score)
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
