
from .base import BaseKVCachePolicy  # 你们框架里的基类
from collections import defaultdict, deque, OrderedDict
import heapq
from typing import Iterable, Dict, List, Deque, Tuple, Optional

class LRUKKVCachePolicy(BaseKVCachePolicy):
    name = "lruk"
    miss: int = 0
    access_count: int = 0

    def __init__(self, cache_size: int, k: int = 8):
        super().__init__(cache_size)
        assert k >= 1
        self.k = k

    def init(self) -> None:
        # 全局时钟/统计
        self.ticks: int = 0
        self.access_count = 0
        self.miss = 0

        # in-cache 标记
        self.cache: set[int] = set()

        # young：历史 < K 次 的在册元素（FIFO，左旧右新）
        self.young: OrderedDict[int, None] = OrderedDict()

        # mature：历史 ≥ K 次 的在册元素
        #   最小堆条目：(kth_time, seq, key) 惰性删除；seq 防止 (t, key) 完全相同导致比较失败
        self._heap: List[Tuple[int, int, int]] = []
        self._seq: int = 0                    # 打散堆的稳定性
        self.kth_time: Dict[int, int] = {}    # key -> 倒数第K次访问时间（用于校验惰性堆项）

        # 历史访问：key -> 近K次访问时间（右端最新）
        self.hist: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=self.k))

    # ---------- 内部工具 ----------
    def _young_add(self, key: int) -> None:
        self.young[key] = None
        self.young.move_to_end(key, last=True)

    def _young_remove(self, key: int) -> None:
        self.young.pop(key, None)

    def _mature_push(self, key: int) -> None:
        # 仅在 hist[key] 已满 K 次时调用
        t_k = self.hist[key][0]  # deque 满时，最左端即“倒数第K次访问时间”
        self.kth_time[key] = t_k
        self._seq += 1
        heapq.heappush(self._heap, (t_k, self._seq, key))

    def _evict_one(self) -> None:
        # 优先驱逐 young（抗污染），否则从 mature 堆弹出“倒数第K次访问最久”的
        if self.young:
            victim, _ = self.young.popitem(last=False)
            self.cache.remove(victim)
            self.kth_time.pop(victim, None)
            return
        # 弹出堆顶直到命中有效条目
        while self._heap:
            t_k, _, key = heapq.heappop(self._heap)
            if key in self.cache and self.kth_time.get(key) == t_k:
                self.cache.remove(key)
                self.kth_time.pop(key, None)
                return
        # 正常不会走到这里（容量检查保证会驱逐出一个）

    # ---------- 接口 ----------
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict | None = None,
    ) -> bool:
        del prompt_blocks, meta
        self.access_count += 1
        self.ticks += 1

        # 记录访问历史
        dq = self.hist[block_id]
        was_len = len(dq)
        dq.append(self.ticks)  # 右端最新

        if block_id in self.cache:
            # 命中：如果从 <K 达到 K，则从 young 提升到 mature；若已是 mature，则更新堆键
            if was_len < self.k and len(dq) >= self.k:
                self._young_remove(block_id)
                self._mature_push(block_id)
            elif len(dq) >= self.k:
                self._mature_push(block_id)  # 惰性去重
            return True

        # 未命中：必要时先驱逐，再插入；LRU-K 的抗污染通过 young 区实现
        if len(self.cache) >= self.cache_size:
            self._evict_one()

        self.cache.add(block_id)
        if len(dq) >= self.k:
            self._mature_push(block_id)
        else:
            self._young_add(block_id)

        self.miss += 1
        return False
