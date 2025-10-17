from __future__ import annotations
from typing import Dict, Iterable, Optional, List, Tuple
from dataclasses import dataclass
from .base import BaseKVCachePolicy  # 提供 add_block / del_block

INF = 1000000000
class PagedEvictionKVCachePolicy(BaseKVCachePolicy):
    """
    WFA: 频率/新近度/位置 + 未来访问保护 + 冷启动LRU + 按类型自适应权重
    - 评分为“保留分 keep_score”（越大越不应淘汰）；驱逐选择 keep_score 最小者
    - 若候选块在本轮当前位置之后仍会被访问：keep_score *= BIG，强力保护
    - 冷启动（前 WARMUP_GLOBAL 次）或块频次过低时退化为 LRU
    """

    name = "wfa"

    # ----------- 可调参数 -----------
    # 冷启动：全局前 X 次访问直接用 LRU；或块频次 <= FREQ_COLD 时 LRU
    WARMUP_GLOBAL: int = 2000
    FREQ_COLD: int = 1

    # 未来访问强力保护：乘以大系数，等价于“几乎不出队”
    FUTURE_MULT: float = 1e6

    # 位置只在头部更有意义，超过该阈值衰减很快（近似公共前缀长度尺度）
    POS_HEAD_CUTOFF: int = 256

    # 各维度权重（会被 type 覆盖）
    W_FREQ: float = 10000
    W_RECENCY: float = 100
    W_POS: float = 1

    # 统计限制（避免内存长大）
    STATS_GC_MULT: int = 1000

    # -------------- 初始化 --------------
    def init(self) -> None:
        self.cache_size: int = getattr(self, "cache_size", 0)
        self.cache: set[int] = set()
        self.access_count: int = 0
        self.miss: int = 0

        # 统计
        self.last_access: Dict[int, int] = {}      # id -> step
        self.freq: Dict[int, int] = {}             # id -> count
        self.min_pos_seen: Dict[int, int] = {}     # id -> 最小出现位置（跨轮前缀价值）

        # 当前 prompt 的位置信息
        self._cur_prompt_key: Optional[int] = None
        self._pos_map: Dict[int, int] = {}
        self._cur_idx: Optional[int] = None

    # ---------- 内部：工具 ----------
    def _ensure_pos_map(self, prompt_blocks: Iterable[int]) -> None:
        seq = list(prompt_blocks)  # 确保顺序，若上游传 set 会破坏语义
        key = hash(tuple(seq))
        if key != self._cur_prompt_key:
            self._cur_prompt_key = key
            self._pos_map = {b: i for i, b in enumerate(seq)}
            self._cur_idx = None

    def _will_appear_later(self, bid: int) -> bool:
        """是否在本轮当前位置之后仍会被访问"""
        if self._cur_idx is None:
            return False
        pos = self._pos_map.get(bid)
        return (pos is not None) and (pos > self._cur_idx)

    def _pos_score(self, bid: int) -> float:
        """
        位置贡献：越靠前越大（越应保留）。
        优先用当前轮位置；若无则用跨轮最小位置 min_pos_seen。
        对超过 POS_HEAD_CUTOFF 的位置快速衰减。
        """
        pos_now = self._pos_map.get(bid)
        pos = pos_now if pos_now is not None else self.min_pos_seen.get(bid, INF)
        if pos >= INF:
            return 0.0
        # 映射到 (0,1]，头部提升
        if pos <= self.POS_HEAD_CUTOFF:
            return 1.0 / (1.0 + pos)
        else:
            # 超过头部区间后迅速衰减（避免远端位置放大作用）
            return 1.0 / (1.0 + self.POS_HEAD_CUTOFF + 4.0 * (pos - self.POS_HEAD_CUTOFF))

    def _recency_score(self, bid: int, now: int) -> float:
        """新近度：越新越大（越应保留），取 1/(age+1) 保持有界"""
        age = now - self.last_access.get(bid, 0)
        return 1.0 / (1.0 + max(0, age))

    def _freq_score(self, bid: int) -> float:
        """频率：取 log(1+f) 平滑放大，防止大数‘碾压’其他维度"""
        f = self.freq.get(bid, 0)
        # Python 无内建 log1p？有的
        import math
        return math.log1p(float(f))

    def _select_weights_by_type(self, meta: Optional[Dict[str, Any]]) -> tuple[float, float, float]:
        """
        根据 prompt 类型微调权重：
        - chat/agent：更强调位置（公共前缀重用）
        - search/rag/docqa：更强调频率（跨轮复用取决于被检索片段热度）
        - 其他：使用默认
        """
        t = None
        if meta:
            t = (meta.get("type") or meta.get("app") or meta.get("prompt_type"))
            if isinstance(t, str):
                t = t.lower().strip()

        if t in {"chat", "agent", "assistant"}:
            return (1.0, 0.6, 1.8)     # freq, rec, pos
        if t in {"search", "rag", "docqa", "qa"}:
            return (1.6, 0.6, 0.9)
        # 默认
        return (self.W_FREQ, self.W_RECENCY, self.W_POS)

    def _keep_score(self, bid: int, now: int, meta: Optional[Dict[str, Any]]) -> float:
        """综合得分：越大越该保留；若本轮稍后还会访问 -> 乘以 FUTURE_MULT 强力保护"""
        wf, wr, wp = self._select_weights_by_type(meta)
        s = (
            wf * self._freq_score(bid)
            + wr * self._recency_score(bid, now)
            + wp * self._pos_score(bid)
        )
        if self._will_appear_later(bid):
            s *= self.FUTURE_MULT
        return s

    def _pick_victim(self, meta: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        选牺牲者：
        - 冷启动或低频：优先 LRU，但仍避免“误杀未来要用”的块（若都将再用，退化为“未来最晚”的）
        - 正常阶段：选 keep_score 最小者
        """
        if not self.cache:
            return None

        now = self.access_count

        # 1) 冷启动 / 低频：LRU with future-guard
        if self.access_count <= self.WARMUP_GLOBAL:
            # 先选“本轮不会再访问”的里最老的；若全都要再访问，选“未来距离最远”的
            oldest_id, oldest_age = None, -1
            furthest_id, furthest_d = None, -1
            for b in self.cache:
                pos = self._pos_map.get(b)
                if (self._cur_idx is None) or (pos is None) or (pos <= self._cur_idx):
                    # 本轮不会再访问
                    age = now - self.last_access.get(b, 0)
                    if age > oldest_age:
                        oldest_age, oldest_id = age, b
                else:
                    d = pos - self._cur_idx
                    if d > furthest_d:
                        furthest_d, furthest_id = d, b
            return oldest_id if oldest_id is not None else furthest_id

        # 2) 正常阶段：按 keep_score
        best_id, best_score = None, None
        for b in self.cache:
            # 若块频次仍很低，可给 LRU 更大话语权（但仍受未来保护）
            if self.freq.get(b, 0) <= self.FREQ_COLD:
                # LRU keep 分（越新越大；若未来还用则乘以 FUTURE_MULT）
                ks = self._recency_score(b, now)
                if self._will_appear_later(b):
                    ks *= self.FUTURE_MULT
            else:
                ks = self._keep_score(b, now, meta)

            if (best_score is None) or (ks < best_score):
                best_score, best_id = ks, b
        return best_id

    # ---------------- 主流程 ----------------
    def access(self, block_id: int, prompt_blocks: Iterable[int], meta: Optional[Dict[str, Any]] = None) -> bool:
        self.access_count += 1

        # 解析当前 prompt 的位置信息 & 更新当前位置
        self._ensure_pos_map(prompt_blocks)
        self._cur_idx = self._pos_map.get(block_id, self._cur_idx)

        # 命中
        if block_id in self.cache:
            self.last_access[block_id] = self.access_count
            self.freq[block_id] = self.freq.get(block_id, 0) + 1
            # 跨轮：维护最小出现位置（近似“公共前缀价值”）
            if self._cur_idx is not None:
                old = self.min_pos_seen.get(block_id, INF)
                if self._cur_idx < old:
                    self.min_pos_seen[block_id] = self._cur_idx
            return True

        # 未命中：必要时驱逐
        if len(self.cache) >= self.cache_size > 0:
            victim = self._pick_victim(meta)
            if victim is not None and victim in self.cache:
                self.cache.remove(victim)
                # 兼容不同环境的删除回调名
                #for name in ("del", "delete", "evict", "remove"):
                #    self._try_call(name, victim)

        # 加载新块（评测规则：所有 miss 都必须 add）
        self.cache.add(block_id)
        self.miss += 1
        self.last_access[block_id] = self.access_count
        self.freq[block_id] = self.freq.get(block_id, 0) + 1
        if self._cur_idx is not None:
            self.min_pos_seen[block_id] = min(self.min_pos_seen.get(block_id, INF), self._cur_idx)
        #self._try_call("add", block_id)

        # 轻量 GC，避免 dict 无界增长
        #limit = self.STATS_GC_MULT * max(1, self.cache_size)
        #if len(self.freq) > limit or len(self.last_access) > limit or len(self.min_pos_seen) > limit:
        #    keep = set(self.cache)
        #    # 额外保留最近/最频繁的一小撮键
        #    for d in (self.last_access, self.freq):
        #        for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[: (limit // 2)]:
        #            keep.add(k)
        #    for d in (self.freq, self.last_access, self.min_pos_seen):
        #        for k in list(d.keys()):
        #            if k not in keep:
        #                d.pop(k, None)

        return False