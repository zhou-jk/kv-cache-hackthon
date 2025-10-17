import heapq
import math
from collections import defaultdict
from typing import Iterable, Dict, Any, Optional, Set, Tuple, List
from .base import BaseKVCachePolicy

import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable, Optional, List, Any

@dataclass
class _BlockEntry:
    block_id: int
    cat: Tuple[str, int]      # (req_type, turn)
    last_tick: int            # last access "time" (以访问次数为时钟)
    offset: float             # 0.0=head, 1.0=tail（在本次 prompt 中的位置归一化）
    size: int = 1             # 以 block 为单位；若有字节数可在meta里覆盖

@dataclass
class _WorkloadStats:
    # 以指数分布拟合复用时间，lam=1/mean_reuse
    mu: float = 60.0          # 复用间隔均值（EWMA，单位：tick）
    lam: float = 1.0 / 60.0   # 指数分布参数
    n_samples: int = 0
    pq: List[Tuple[int, int]] = field(default_factory=list)  # (last_tick, block_id)，最旧在堆顶
    blocks: set = field(default_factory=set)                 # 该类下的 block 集合

    def life(self, life_cap: float, life_mult: float = 1.0) -> float:
        # 论文中 life 用于限制时间窗，避免长尾（表2➂）。用 E[T]=1/lam 作为默认期望寿命，再设 cap。
        expected = max(1.0 / self.lam, 1.0)
        return min(expected * life_mult, life_cap)

    def update_reuse(self, dt: float, alpha: float = 0.1) -> None:
        if dt <= 0:
            return
        if self.n_samples == 0:
            self.mu = dt
        else:
            self.mu = (1 - alpha) * self.mu + alpha * dt
        self.lam = 1.0 / max(self.mu, 1e-6)
        self.n_samples += 1


class WorkloadAwareKVCachePolicy(BaseKVCachePolicy):
    """
    论文 §4.2 的“工作负载感知 + 复用概率分布”淘汰：优先级 = (ReuseProb_w(t, life), -Offset)
    - 为追求最高命中率，默认使用 global 精确扫描 O(N) 选受害者；
      如需低开销可设置 fast=True 启用论文里的 O(W) 优化（每类仅取一个候选）。
    - 评测要求：每次 access 后 block 必须在 cache；每次 add 计 miss。
    """
    name = "wa"  # workload-aware

    def __init__(
        self,
        cache_size: int,
        *,
        fast: bool = True,     # False=全局精确；True=论文优化 O(W)
        alpha: float = 0.1,     # EWMA 更新速度（拟合指数分布的均值）
        life_cap: int = 512,    # 限制 life 的时间窗（tick）
        life_mult: float = 1.0  # life = min(E[T]*life_mult, life_cap)
    ):
        super().__init__(cache_size)
        self.fast = fast
        self.alpha = alpha
        self.life_cap = life_cap
        self.life_mult = life_mult

    # ---------------- Framework Hooks ----------------
    def init(self) -> None:
        self.ticks: int = 0
        self.access_count: int = 0
        self.miss: int = 0

        self.cache: Dict[int, _BlockEntry] = {}
        self.workloads: Dict[Tuple[str, int], _WorkloadStats] = {}

        # 评测环境通常提供 add()/del()；若无则退化为空操作
        self._add_hook = getattr(self, "add", lambda _id: None)
        self._del_hook = getattr(self, "del_", getattr(self, "delete", lambda _id: None))

    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        返回 True=hit, False=miss（每次 add 记一次 miss）。调用结束后，block 一定在 cache 中。
        """
        self.access_count += 1
        self.ticks += 1

        cat = self._category_from_meta(meta)
        stats = self._get_or_create_stats(cat)
        offset = self._position_offset(block_id, prompt_blocks)

        # --- Hit path ---
        if block_id in self.cache:
            ent = self.cache[block_id]
            dt = self.ticks - ent.last_tick
            if dt > 0:
                stats.update_reuse(dt, self.alpha)  # 采样复用时间，用于拟合 lam
            # 刷新元数据
            ent.last_tick = self.ticks
            ent.offset = offset
            heapq.heappush(stats.pq, (ent.last_tick, block_id))  # 惰性去重
            return True

        # --- Miss path: 需要逐出直到有空位 ---
        while len(self.cache) >= self.cache_size:
            victim = self._choose_victim()
            if victim is None:
                break
            v_ent = self.cache.pop(victim, None)
            if v_ent is not None:
                v_stats = self.workloads[v_ent.cat]
                v_stats.blocks.discard(victim)
            self._del_hook(victim)

        # 插入并记 miss
        ent = _BlockEntry(block_id=block_id, cat=cat, last_tick=self.ticks, offset=offset)
        self.cache[block_id] = ent
        stats.blocks.add(block_id)
        heapq.heappush(stats.pq, (ent.last_tick, block_id))
        self._add_hook(block_id)
        self.miss += 1
        return False

    # ---------------- Internals ----------------
    def _category_from_meta(self, meta: Optional[Dict[str, Any]]) -> Tuple[str, int]:
        """
        工作负载类别： (req_type, turn)
        - req_type 优先从 meta['type'/'req_type'/'app_type'] 取；默认 'unknown'
        - turn 优先从 meta['turn'/'turn_id'/'round'/'session_turn'] 取；默认 1
        论文指出“按类别（type×turn）概率分布可预测且稳定”（图 11、15，页 6–7）。:contentReference[oaicite:1]{index=1}
        """
        if meta is None:
            return ("unknown", 1)
        t = meta.get("type") or meta.get("req_type") or meta.get("app_type") or "unknown"
        turn = meta.get("turn") or meta.get("turn_id") or meta.get("round") or meta.get("session_turn") or 1
        try:
            turn = int(turn)
        except Exception:
            turn = 1
        return (str(t), turn)

    def _get_or_create_stats(self, cat: Tuple[str, int]) -> _WorkloadStats:
        s = self.workloads.get(cat)
        if s is None:
            s = _WorkloadStats(mu=50.0, lam=1.0 / 50.0)
            self.workloads[cat] = s
        return s

    def _position_offset(self, block_id: int, prompt_blocks: Iterable[int]) -> float:
        """
        计算该 block 在当前 prompt 中的归一化位置 [0,1]；0=head，1=tail。
        根据论文的空间局部性分析，应优先保留前缀（图 16–18，页 7–8）。:contentReference[oaicite:2]{index=2}
        """
        try:
            if not isinstance(prompt_blocks, list):
                prompt_blocks = list(prompt_blocks)
            n = len(prompt_blocks)
            if n == 0:
                return 1.0
            idx = prompt_blocks.index(block_id)  # 若有重复，取第一次出现（通常 block 唯一）
            return idx / max(n - 1, 1)
        except Exception:
            return 1.0  # 无法判断时按 tail 处理

    @staticmethod
    def _reuse_prob(lam: float, age: float, life: float) -> float:
        """
        指数分布 CDF: F(x) = 1 - e^{-lam x}
        ReuseProb(t, life) = F(t+life) - F(t) = e^{-lam t} * (1 - e^{-lam life})
        对应论文图 23 的公式（页 11）。:contentReference[oaicite:3]{index=3}
        """
        if life <= 0:
            return 0.0
        return math.exp(-lam * max(age, 0.0)) * (1.0 - math.exp(-lam * life))

    def _choose_victim(self) -> Optional[int]:
        if not self.cache:
            return None
        return self._eviction_candidate_fast() if self.fast else self._eviction_candidate_global()

    def _eviction_candidate_global(self) -> int:
        """
        O(N) 精确：遍历 cache 中所有 block，按优先级 (prob, -offset) 取最小的逐出。
        为求最高命中率，默认使用该模式。
        """
        cur = self.ticks
        victim = None
        best_priority = None
        for bid, ent in self.cache.items():
            stats = self.workloads[ent.cat]
            age = cur - ent.last_tick
            life = stats.life(self.life_cap, self.life_mult)
            p = self._reuse_prob(stats.lam, age, life)
            priority = (p, -ent.offset)  # 概率越小越先逐出；offset 越大（越靠尾部）越先逐出
            if best_priority is None or priority < best_priority:
                best_priority = priority
                victim = bid
        return victim

    def _eviction_candidate_fast(self) -> int:
        """
        论文的 O(W) 优化：每个工作负载类别取 LRU 候选，再在这些候选上比较概率。
        见 §4.2“Performance optimization”（页 11）。:contentReference[oaicite:4]{index=4}
        """
        cur = self.ticks
        victim = None
        best_priority = None
        for cat, stats in self.workloads.items():
            # 堆惰性去重：直到堆顶记录与当前缓存一致
            while stats.pq:
                t, bid = stats.pq[0]
                ent = self.cache.get(bid)
                if ent is None or ent.last_tick != t:
                    heapq.heappop(stats.pq)
                    if ent is not None:
                        heapq.heappush(stats.pq, (ent.last_tick, bid))
                    continue
                break
            if not stats.pq:
                continue
            _, bid = stats.pq[0]
            ent = self.cache.get(bid)
            if ent is None:
                continue
            age = cur - ent.last_tick
            life = stats.life(self.life_cap, self.life_mult)
            p = self._reuse_prob(stats.lam, age, life)
            priority = (p, -ent.offset)
            if best_priority is None or priority < best_priority:
                best_priority = priority
                victim = bid
        # 极端情况下回退到全局扫描
        return victim if victim is not None else self._eviction_candidate_global()