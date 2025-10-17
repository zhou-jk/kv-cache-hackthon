from collections import deque, OrderedDict
from typing import Iterable, Dict, Any, Optional, Set, Deque

from .base import BaseKVCachePolicy, BlockEntry, calc_prefix_score

class S3FIFOKVCachePolicy(BaseKVCachePolicy):
    """
    S3-FIFO for KV Cache (faithful implementation)
    - 三条静态 FIFO 队列：S(小队列，默认 10%) / M(主队列，90%) / G(幽灵队列，仅存key)
    - 两位访问计数：0..3，命中时 min(cnt+1, 3)
    - 插入规则：若在 G 中则直接进 M，否则进 S
    - 驱逐规则：
        * 全局满 -> 反复执行 evict() 直到不满
        * evict()：若 |S| >= r*capacity 先从 S 清，否则从 M 清
        * evictS()：弹 S 尾 t，若 t.freq > 1 -> 移入 M（并清零计数）；否则 -> 放入 G（真正释放 1 容量）
        * evictM()：弹 M 尾 t，若 t.freq > 0 -> 头插回 M 且 t.freq -= 1（懒提升）；否则 -> 物理淘汰
    - 与评测器约定：所有 miss 必须 add()，并记一次 miss；内部 S<->M/G 迁移不触发 add/miss
    参考：
      * 论文伪代码 Algorithm 1（S、M、G 及两位计数）
      * 作者博客对三队列/插入/移动/重插的说明（L137-L141 等）
    """

    name = "s3fifo"

    # ---- 可调参数 ----
    SMALL_RATIO: float = 0.10   # |S| 占总容量比例
    FREQ_MAX: int = 3           # 2-bit 饱和计数上限
    # G 的容量 = M 的容量（与论文一致）
    # 其他：不做任何“额外优化”，忠实实现 S3-FIFO

    # 统计
    miss: int = 0
    access_count: int = 0

    # ---------- 初始化 ----------
    def init(self) -> None:
        self.cache_size: int = getattr(self, "cache_size", 0)
        # S/M 队列（真实驻留对象）
        self.S: Deque[int] = deque()
        self.M: Deque[int] = deque()
        # G 幽灵队列：仅记录 key（FIFO），用 OrderedDict 维护顺序+O(1) 删除
        self.G: "OrderedDict[int, None]" = OrderedDict()

        # 成员关系与计数
        self.in_S: Set[int] = set()
        self.in_M: Set[int] = set()
        self.freq: Dict[int, int] = {}   # 0..FREQ_MAX

        # 容量（静态）
        self.S_cap: int = int(self.SMALL_RATIO * self.cache_size)
        self.M_cap: int = max(0, self.cache_size - self.S_cap)
        self.G_cap: int = self.M_cap  # 论文：G 的 ghost entries 数量与 M 相同 

        # 其他状态
        self.miss = 0
        self.access_count = 0

    # ---------- 后端 add/del ----------
    @staticmethod
    def _try_call_backend(fn_name: str, block_id: int) -> None:
        fn = globals().get(fn_name)
        if callable(fn):
            try:
                fn(block_id)
            except Exception:
                pass

    def _backend_add(self, block_id: int) -> None:
        # 评测要求：miss 必须 add
        self._try_call_backend("add", block_id)

    def _backend_del(self, block_id: int) -> None:
        # 兼容可能的不同删除函数名
        for name in ("delete", "evict", "remove"):
            fn = globals().get(name)
            if callable(fn):
                try:
                    fn(block_id)
                    return
                except Exception:
                    pass
        self._try_call_backend("del", block_id)

    # ---------- 工具 ----------
    def _cache_full(self) -> bool:
        return (len(self.S) + len(self.M)) >= self.cache_size > 0

    def _in_cache(self, bid: int) -> bool:
        return (bid in self.in_S) or (bid in self.in_M)

    # ---------- G（幽灵队列）维护 ----------
    def _ghost_has(self, bid: int) -> bool:
        return bid in self.G

    def _ghost_add(self, bid: int) -> None:
        # 放到队尾表示“最新”；超限 pop 最旧（队首）
        if bid in self.G:
            self.G.move_to_end(bid, last=True)
        else:
            self.G[bid] = None
        if len(self.G) > self.G_cap:
            self.G.popitem(last=False)  # FIFO: 弹最老

    def _ghost_remove(self, bid: int) -> None:
        if bid in self.G:
            try:
                del self.G[bid]
            except Exception:
                pass

    # ---------- 频次 ----------
    def _hit_bump(self, bid: int) -> None:
        self.freq[bid] = min(self.freq.get(bid, 0) + 1, self.FREQ_MAX)

    # ---------- 驱逐 ----------
    def _evict(self) -> None:
        """
        全局驱逐：让 |S| 靠近 SMALL_RATIO*capacity
        - 若 |S| 已达上限 -> 优先从 S 做一次处理（可能把 S 尾移入 M 或 G）
        - 否则从 M 做一次处理（M 尾 reinsert 或真正淘汰）
        """
        if len(self.S) >= self.S_cap:
            self._evictS()
        else:
            self._evictM()

    def _evictS(self) -> None:
        """
        参考 Algo.1 evictS：
          取 S 尾 t：
            if t.freq > 1: 迁入 M（清零计数）
              若 |M|==M_cap -> 先 evictM()
            else: 放入 G（真正释放 1 个容量）
          无论哪种情况，t 都从 S 删除
        """
        evicted = False
        while (not evicted) and self.S:
            t = self.S[-1]  # 先看尾
            f = self.freq.get(t, 0)
            # 从 S 删除（无论迁往 M 还是入 G）
            self.S.pop()
            self.in_S.discard(t)

            if f > 1:
                # 迁入 M（可能需要先挪走 M 尾）
                while len(self.M) >= self.M_cap and (self.M_cap > 0):
                    self._evictM()
                    # 这里 _evictM() 一次一定会释放 1 个容量
                # 头插入 M，并清零计数（论文文本明确“移动时清零”）
                self.M.appendleft(t)
                self.in_M.add(t)
                self.freq[t] = 0
                # 注意：本分支没有真正释放容量（S->M），故 evicted=False，继续循环
            else:
                # 放入幽灵队列（释放 1 容量）
                self._ghost_add(t)
                # 真正物理删除（确保 KV 从 DRAM 中移除）
                self._backend_del(t)
                self.freq.pop(t, None)
                evicted = True

    def _evictM(self) -> None:
        """
        参考 Algo.1 evictM：
          取 M 尾 t：
            if t.freq > 0: 头插回 M，并 t.freq -= 1（懒提升 / reinsertion）
            else: 真正物理淘汰（释放 1 容量）
        """
        evicted = False
        while (not evicted) and self.M:
            t = self.M[-1]
            f = self.freq.get(t, 0)
            if f > 0:
                # 懒提升：把尾部对象重新放回头部，频次-1
                self.M.pop()
                self.M.appendleft(t)
                self.freq[t] = f - 1
                # 未释放容量，继续尝试
            else:
                # 真正淘汰
                self.M.pop()
                self.in_M.discard(t)
                self._backend_del(t)
                self.freq.pop(t, None)
                evicted = True

    # ---------- 插入 ----------
    def _insert(self, bid: int) -> None:
        """
        参考 Algo.1 insert：
          - 若在 G -> 插到 M 头
          - 否则 -> 插到 S 头
          - 插入前若全局满 -> 不断 _evict()
          - 新插入对象 freq=0
        """
        # 先释放到不满
        while self._cache_full():
            self._evict()

        if self._ghost_has(bid):
            # 命中幽灵：直接进 M，并从 G 移除其 ghost
            self._ghost_remove(bid)
            # 确保 M 不超自己的静态上限
            while len(self.M) >= self.M_cap and (self.M_cap > 0):
                self._evictM()
            self.M.appendleft(bid)
            self.in_M.add(bid)
        else:
            # 正常新对象：进 S
            self.S.appendleft(bid)
            self.in_S.add(bid)

        # 新插入频次清零（论文 Algo.1 第 6 行 / 482-483）
        self.freq[bid] = 0

    # ---------- 访问主流程 ----------
    def access(self, block_id: int, prompt_blocks: Iterable[int], meta: Optional[Dict[str, Any]] = None) -> bool:
        """
        评测约束：
          - 命中：返回 True，不调用 add()
          - 未命中：必须 add() 并 miss++，随后对象在缓存中（由 _insert 完成）；必要时驱逐
        """
        del prompt_blocks, meta  # S3-FIFO 不使用内容/类型特征
        self.access_count += 1

        # 命中：在 S 或 M
        if self._in_cache(block_id):
            self._hit_bump(block_id)  # 两位计数 +1（饱和到 3）
            return True

        # 未命中：写入并计 miss
        self.miss += 1
        self._backend_add(block_id)  # 评测要求：add() 表示一次 miss 的加载
        self._insert(block_id)
        return False
