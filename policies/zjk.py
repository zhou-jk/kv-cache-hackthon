from __future__ import annotations
from typing import Dict, Iterable, Optional, List, Tuple
from dataclasses import dataclass
from .base import BaseKVCachePolicy

import heapq
import math
import heapq
import math
from typing import Iterable, Dict, Any, Optional, List, Tuple

INF = 1 << 30

class PagedEvictionKVCachePolicy(BaseKVCachePolicy):
    """
    WFA-Heap: 频率/新近度/位置 + 未来访问保护 + 冷启动LRU + type加权
    受害者选择使用小根堆 + 懒失效：
      - cand_heap 存 (keep_score, gen, block_id)，keep_score越小越先淘
      - stamp[block_id] 记录该块当前有效的“代数”；堆顶条目若代数过期或块已被驱逐则丢弃
      - 需要时对堆顶块“现时重算”分数；若变化则以新分数入堆并继续弹出下一项
    其它策略保持不变（不要添加额外优化）。
    """

    name = "wfa_heap"

    # ---------- 可调参数（与原 WFA 相同） ----------
    WARMUP_GLOBAL: int = 2000      # 冷启动：全局前 X 次访问用 LRU（含未来保护）
    FREQ_COLD: int = 1             # 单块频次<=该阈值时，频率弱，倾向 LRU
    FUTURE_MULT: float = 1e6       # 未来将再用的强力保护：keep_score *= 该系数
    POS_HEAD_CUTOFF: int = 256     # 位置的“头部”区间（前缀更有价值）
    W_FREQ: float = 1.0
    W_RECENCY: float = 0.6
    W_POS: float = 1.2
    STATS_GC_MULT: int = 1000         # 统计结构的上限系数，避免无界增长
    HEAP_EPS: float = 1e-9         # 分数重算时的容差

    # 统计
    miss: int = 0
    access_count: int = 0

    # -------------- 初始化 --------------
    def init(self) -> None:
        self.cache_size: int = getattr(self, "cache_size", 0)
        self.cache: set[int] = set()
        self.access_count = 0
        self.miss = 0

        # 频率/新近度/跨轮前缀价值
        self.last_access: Dict[int, int] = {}
        self.freq: Dict[int, int] = {}
        self.min_pos_seen: Dict[int, int] = {}

        # 当前 prompt 的位置信息
        self._cur_prompt_key: Optional[int] = None
        self._pos_map: Dict[int, int] = {}
        self._cur_idx: Optional[int] = None
        self._cur_prompt_id: int = 0

        # ---- 小根堆 + 懒失效 ----
        self._cand_heap: List[Tuple[float, int, int]] = []  # (keep_score, gen, id)
        self._stamp: Dict[int, int] = {}                   # id -> 当前有效代数
        self._gen: int = 0

    # ---------- 工具：位置映射 ----------
    def _ensure_pos_map(self, prompt_blocks: Iterable[int]) -> None:
        seq = list(prompt_blocks)  # 必须有序；若上游传 set 会破坏位置/未来语义
        key = hash(tuple(seq))
        if key != self._cur_prompt_key:
            self._cur_prompt_key = key
            self._pos_map = {b: i for i, b in enumerate(seq)}
            self._cur_idx = None
            self._cur_prompt_id += 1

    # ---------- 工具：未来、本轮位置 ----------
    def _will_appear_later(self, bid: int) -> bool:
        if self._cur_idx is None:
            return False
        pos = self._pos_map.get(bid)
        return (pos is not None) and (pos > self._cur_idx)

    # ---------- 三个维度的分数 ----------
    def _pos_score(self, bid: int) -> float:
        pos_now = self._pos_map.get(bid)
        pos = pos_now if pos_now is not None else self.min_pos_seen.get(bid, INF)
        if pos >= INF:
            return 0.0
        if pos <= self.POS_HEAD_CUTOFF:
            return 1.0 / (1.0 + pos)
        return 1.0 / (1.0 + self.POS_HEAD_CUTOFF + 4.0 * (pos - self.POS_HEAD_CUTOFF))

    def _recency_score(self, bid: int, now: int) -> float:
        age = now - self.last_access.get(bid, 0)
        return 1.0 / (1.0 + max(0, age))

    def _freq_score(self, bid: int) -> float:
        f = self.freq.get(bid, 0)
        return math.log1p(float(f))

    # ---------- type 影响权重 ----------
    def _select_weights_by_type(self, meta: Optional[Dict[str, Any]]) -> tuple[float, float, float]:
        t = None
        if meta:
            t = (meta.get("type") or meta.get("app") or meta.get("prompt_type"))
            if isinstance(t, str):
                t = t.lower().strip()
        if t in {"chat", "agent", "assistant"}:
            return (1.0, 0.6, 1.8)    # 更看重前缀位置
        if t in {"search", "rag", "docqa", "qa"}:
            return (1.6, 0.6, 0.9)    # 更看重频率
        return (self.W_FREQ, self.W_RECENCY, self.W_POS)

    # ---------- 组合成 keep_score（越大越该保留） ----------
    def _keep_score_full(self, bid: int, now: int, meta: Optional[Dict[str, Any]]) -> float:
        wf, wr, wp = self._select_weights_by_type(meta)
        s = wf * self._freq_score(bid) + wr * self._recency_score(bid, now) + wp * self._pos_score(bid)
        if self._will_appear_later(bid):
            s *= self.FUTURE_MULT
        return s

    def _keep_score_for_heap(self, bid: int, now: int, meta: Optional[Dict[str, Any]]) -> float:
        # 频次还很低时更像 LRU（但仍受未来保护）
        if self.freq.get(bid, 0) <= self.FREQ_COLD:
            s = self._recency_score(bid, now)
            if self._will_appear_later(bid):
                s *= self.FUTURE_MULT
            return s
        return self._keep_score_full(bid, now, meta)

    # ---------- 小根堆：入堆（带代数） ----------
    def _heap_push(self, bid: int, meta: Optional[Dict[str, Any]]) -> None:
        self._gen += 1
        self._stamp[bid] = self._gen
        now = self.access_count
        ks = self._keep_score_for_heap(bid, now, meta)
        heapq.heappush(self._cand_heap, (ks, self._gen, bid))

    # ---------- 小根堆：选受害者（懒失效 + 现时重算） ----------
    def _pick_victim_heap(self, meta: Optional[Dict[str, Any]]) -> Optional[int]:
        # 冷启动：仍用 LRU（但避免误杀“未来即将访问”的块）
        now = self.access_count
        if self.access_count <= self.WARMUP_GLOBAL:
            oldest_id, oldest_age = None, -1
            furthest_id, furthest_d = None, -1
            for b in self.cache:
                pos = self._pos_map.get(b)
                if (self._cur_idx is None) or (pos is None) or (pos <= self._cur_idx):
                    age = now - self.last_access.get(b, 0)
                    if age > oldest_age:
                        oldest_age, oldest_id = age, b
                else:
                    d = pos - self._cur_idx
                    if d > furthest_d:
                        furthest_d, furthest_id = d, b
            return oldest_id if oldest_id is not None else furthest_id

        # 正常阶段：从堆顶开始懒失效
        while self._cand_heap:
            ks_old, gen_old, bid = heapq.heappop(self._cand_heap)
            # 已被驱逐 / 代数过期 → 丢弃
            if (bid not in self.cache) or (self._stamp.get(bid) != gen_old):
                continue
            # 现时重算分数，若变化则再次入堆
            ks_now = self._keep_score_for_heap(bid, now, meta)
            if abs(ks_now - ks_old) > self.HEAP_EPS:
                self._gen += 1
                self._stamp[bid] = self._gen
                heapq.heappush(self._cand_heap, (ks_now, self._gen, bid))
                continue
            # 分数仍然有效 → 选为受害者
            return bid

        # 极端情况：堆空（例如全是过期条目），退化到简单 LRU（带未来保护）
        oldest_id, oldest_age = None, -1
        furthest_id, furthest_d = None, -1
        for b in self.cache:
            pos = self._pos_map.get(b)
            if (self._cur_idx is None) or (pos is None) or (pos <= self._cur_idx):
                age = now - self.last_access.get(b, 0)
                if age > oldest_age:
                    oldest_age, oldest_id = age, b
            else:
                d = pos - self._cur_idx
                if d > furthest_d:
                    furthest_d, furthest_id = d, b
        return oldest_id if oldest_id is not None else furthest_id

    # ---------- 后端回调 ----------
    @staticmethod
    def _try_call(fn_name: str, bid: int) -> None:
        fn = globals().get(fn_name)
        if callable(fn):
            try:
                fn(bid)
            except Exception:
                pass

    # ---------------- 主流程 ----------------
    def access(self, block_id: int, prompt_blocks: Iterable[int], meta: Optional[Dict[str, Any]] = None) -> bool:
        self.access_count += 1

        # 解析/更新当前位置索引
        self._ensure_pos_map(prompt_blocks)
        self._cur_idx = self._pos_map.get(block_id, self._cur_idx)

        # 命中
        if block_id in self.cache:
            self.last_access[block_id] = self._cur_prompt_id
            self.freq[block_id] = self.freq.get(block_id, 0) + 1
            if self._cur_idx is not None:
                old = self.min_pos_seen.get(block_id, INF)
                if self._cur_idx < old:
                    self.min_pos_seen[block_id] = self._cur_idx
            # 命中也入堆一份新快照（懒失效覆盖旧分数）
            self._heap_push(block_id, meta)
            return True

        # 未命中：必要时驱逐
        if len(self.cache) >= self.cache_size > 0:
            victim = self._pick_victim_heap(meta)
            if victim is not None and victim in self.cache:
                self.cache.remove(victim)
                for name in ("delete", "evict", "remove", "del"):
                    self._try_call(name, victim)

        # 加载新块（评测规则：所有 miss 都要 add，并计一次 miss）
        self.cache.add(block_id)
        self.miss += 1
        self.last_access[block_id] = self.access_count
        self.freq[block_id] = self.freq.get(block_id, 0) + 1
        if self._cur_idx is not None:
            self.min_pos_seen[block_id] = min(self.min_pos_seen.get(block_id, INF), self._cur_idx)
        self._try_call("add", block_id)

        # 新块入堆
        self._heap_push(block_id, meta)

        # 轻量 GC：防止统计结构无界增长
        limit = self.STATS_GC_MULT * max(1, self.cache_size)
        if len(self.freq) > limit or len(self.last_access) > limit or len(self.min_pos_seen) > limit:
            keep = set(self.cache)
            # 额外保留最近/最频繁的一撮键，避免清掉活跃的统计
            for d in (self.last_access, self.freq):
                for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[: (limit // 2)]:
                    keep.add(k)
            for d in (self.freq, self.last_access, self.min_pos_seen):
                for k in list(d.keys()):
                    if k not in keep:
                        d.pop(k, None)

        return False
