from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterable, Set, Tuple
from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score

from typing import Iterable, Dict, List, Any, Optional, Tuple
import math, heapq

from typing import Iterable, Dict, Any, Optional, Tuple
from collections import OrderedDict, defaultdict
import math

class WorkloadAwareKVCachePolicy(BaseKVCachePolicy):
    name = "workload_aware_kvcache"
    miss: int = 0
    access_count: int = 0

    PRIOR_MEAN_DT = 300.0
    EWMA_ALPHA    = 0.1
    LIFE_CAP      = 3600.0
    MIN_LAMBDA    = 1e-6
    LIFE_MULT_C   = 4.0   # life = C / λ

    def init(self) -> None:
        self.cache: set[int] = set()
        self.cache_size = max(int(self.cache_size), 1)

        self.t = 0
        self.miss = 0
        self.access_count = 0

        self.last_access: Dict[int, int] = {}
        self.block_offset: Dict[int, float] = {}
        self.block_workload: Dict[int, str] = {}
        self.w_mean_dt: Dict[str, float] = {}
        self.w_count: Dict[str, int] = {}
        self.w_lru: Dict[str, OrderedDict[int, None]] = defaultdict(OrderedDict)

    # ---- helpers ----
    def _workload_key(self, meta: Optional[Dict[str, Any]]) -> str:
        if not meta: return "unknown-1"
        return f"{str(meta.get('type','unknown'))}-{int(meta.get('turn',1))}"

    def _offset_ratio(self, block_id: int, prompt: list[int]) -> float:
        try:
            idx = prompt.index(block_id); n = max(1, len(prompt))
            return idx / n
        except ValueError:
            return 1.0

    def _update_reuse_stats(self, bid: int, w: str) -> None:
        prev = self.last_access.get(bid)
        if prev is None: return
        dt = max(1, self.t - prev)
        mean_dt = self.w_mean_dt.get(w, self.PRIOR_MEAN_DT)
        self.w_mean_dt[w] = (1 - self.EWMA_ALPHA) * mean_dt + self.EWMA_ALPHA * float(dt)
        self.w_count[w] = self.w_count.get(w, 0) + 1

    def _lambda_of(self, w: str) -> float:
        mean_dt = self.w_mean_dt.get(w, self.PRIOR_MEAN_DT)
        return max(1.0/mean_dt if mean_dt>0 else 1.0/self.PRIOR_MEAN_DT, self.MIN_LAMBDA)

    def _life_window(self, w: str) -> float:
        lam = self._lambda_of(w)
        return min(self.LIFE_MULT_C / lam, self.LIFE_CAP)

    def _reuse_prob(self, w: str, age: float) -> float:
        lam = self._lambda_of(w); life = self._life_window(w)
        e1 = math.exp(-lam * max(0.0, age)); e2 = math.exp(-lam * max(0.0, age + life))
        return max(0.0, e1 - e2)

    def _touch_lru(self, w: str, bid: int, ensure_present: bool) -> None:
        lru = self.w_lru[w]
        if bid in lru: lru.move_to_end(bid, last=True)  # O(1)
        elif ensure_present: lru[bid] = None

    def _choose_victim(self) -> Optional[int]:
        best = None  # (prob, -offset, bid, workload)
        for w, lru in self.w_lru.items():
            if not lru: continue
            head_bid = next(iter(lru))  # peek LRU 头，均摊 O(1)
            if head_bid not in self.cache:
                lru.pop(head_bid, None); continue
            age = float(self.t - self.last_access.get(head_bid, self.t))
            p = self._reuse_prob(w, age)
            neg_off = -float(self.block_offset.get(head_bid, 1.0))
            cand = (p, neg_off, head_bid, w)
            if (best is None) or (cand < best): best = cand
        if best is None: return None
        _, _, victim, wstar = best
        self.w_lru[wstar].pop(victim, None)  # 从该 workload LRU 中移除
        return victim

    # ---- API ----
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        self.access_count += 1
        self.t += 1

        prompt = list(prompt_blocks)
        w = self._workload_key(meta)

        # 在线估计复用时间
        self._update_reuse_stats(block_id, w)

        # 更新时间与位置信息
        self.last_access[block_id] = self.t
        if block_id not in self.block_offset:
            self.block_offset[block_id] = self._offset_ratio(block_id, prompt)
        self.block_workload[block_id] = w

        # 命中：只刷新该 workload 的 LRU
        if block_id in self.cache:
            self._touch_lru(w, block_id, ensure_present=False)
            return True

        # 未命中：如满则 O(W) 选牺牲者
        if len(self.cache) >= self.cache_size:
            victim = self._choose_victim()
            if victim is not None:
                self.cache.discard(victim)

        # 极端保护（几乎不会）
        if len(self.cache) >= self.cache_size and self.cache:
            oldest = min(self.cache, key=lambda b: self.last_access.get(b, -1))
            self.cache.discard(oldest)
            w_old = self.block_workload.get(oldest)
            if w_old is not None:
                self.w_lru[w_old].pop(oldest, None)

        # === 真正 add：此时才计一次 miss ===
        self.cache.add(block_id)
        self._touch_lru(w, block_id, ensure_present=True)
        self.miss += 1
        return False
