"""单节点 GPU KV 缓存策略对比框架。

提供多种策略实现（FIFO、LRU、前缀感知、增强版）并支持直接对比。
所有策略遵循相同的 `init` / `access` 接口，便于在相同 trace 上评估命中率。
"""

from __future__ import annotations
import argparse
import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import math

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
    """
    PALFU: Prefix-Aware LFU/LRU Fusion Policy
    - Prefill-only, block-level KV cache policy.
    - Core: prefix pinning + (frequency + recency + id-bias) score + 2-stage eviction.
    """
    name = "palfu"

    # ------------------------ 可调参数 ------------------------
    # 预留给“前缀”的比例上限（占总 cache 的份额），0.0~0.9 之间较合理
    PREFIX_RESERVE_RATIO: float = 0.4
    # 前缀长度 EMA 的平滑系数（值越大越保守）
    PREFIX_EMA_ALPHA: float = 0.9
    # 分值权重：频率、时间新近、编号偏置
    W_FREQ: float = 1.0
    W_RECENCY: float = 0.6
    W_IDBIAS: float = 1.4
    # 统计裁剪阈值（避免 dict 无限增长）
    STATS_GC_MULTIPLIER: int = 8  # 频率/时间字典上限 ~ 8 * cache_size

    def init(self) -> None:
        self.cache: Set[int] = set()
        self.step: int = 0                          # 全局时间步（每次 access 自增）
        self.freq: Dict[int, int] = defaultdict(int)  # 块全局访问次数（LFU）
        self.last_access: Dict[int, int] = {}         # 最近访问时间步（LRU)
        # 自适应前缀估计（EMA）
        self.prefix_len_ema: float = 0.0
        # 最近一次用于计算前缀的 prompt 指纹（避免重复计算）
        self._last_prompt_fp: int | None = None

    # ------------------------ 对接底层add/del（若存在） ------------------------
    @staticmethod
    def _call_backend(fn_name: str, block_id: int) -> None:
        fn = globals().get(fn_name, None)
        if callable(fn):
            try:
                fn(block_id)
            except Exception:
                pass

    def _backend_add(self, block_id: int) -> None:
        # 常见实现叫 add()
        self._call_backend("add", block_id)

    def _backend_del(self, block_id: int) -> None:
        # 有的环境叫 delete/evict/remove，这里都尝试一下
        for name in ("delete", "evict", "remove"):
            fn = globals().get(name, None)
            if callable(fn):
                try:
                    fn(block_id)
                    return
                except Exception:
                    pass
        # 若运行环境恰好提供了名为 del 的函数（极少见），也做保护性尝试
        self._call_backend("del", block_id)

    # ------------------------ 前缀长度估计 ------------------------
    @staticmethod
    def _contiguous_prefix_len(blocks: Iterable[int]) -> int:
        """
        计算从 1 开始的“连续前缀长度”：
        例如 {1,2,3,5,8} -> 3；{2,3,4} -> 0；{1,2,6} -> 2
        """
        s = set(blocks)
        if 1 not in s:
            return 0
        k = 1
        # 在端侧/题目 trace 中，编号一般不离谱；逐步判断足够快
        while (k + 1) in s:
            k += 1
        return k

    def _maybe_update_prefix_ema(self, prompt_blocks: Iterable[int]) -> None:
        """
        只在一个 prompt 的“第一块访问”附近更新一次前缀 EMA，避免反复扫描。
        粗略判定：当访问的 block_id 等于该 prompt 的最小编号（通常是 1）时更新。
        """
        if not prompt_blocks:
            return
        # 指纹（不可逆快速 hash）：max_id ^ len ^ min_id
        blist = list(prompt_blocks)
        mn, mx, ln = min(blist), max(blist), len(blist)
        fp = (mx << 21) ^ (mn << 11) ^ ln
        if self._last_prompt_fp == fp:
            return
        self._last_prompt_fp = fp

        if mn == min(blist):  # “看起来像是新 prompt 的开始”
            prefix_len = self._contiguous_prefix_len(blist)
            self.prefix_len_ema = (
                self.PREFIX_EMA_ALPHA * self.prefix_len_ema
                + (1.0 - self.PREFIX_EMA_ALPHA) * float(prefix_len)
            )

    def _pinned_target_count(self) -> int:
        """
        需要预留给前缀的目标数量：受 cache_size * ratio 与 prefix_ema 双重约束。
        """
        hard_cap = int(self.cache_size * self.PREFIX_RESERVE_RATIO)
        ema_cap = int(self.prefix_len_ema + 1e-6)
        return max(0, min(hard_cap, ema_cap))

    def _current_pinned_set(self) -> Set[int]:
        """
        依据当前 cache 内容与目标前缀长度，确定“本轮不希望驱逐”的 pinned 集合。
        这里的启发式很简单：把 cache 里编号最小的 r 个当作前缀保留（r 为目标数）。
        """
        r = self._pinned_target_count()
        if r <= 0 or not self.cache:
            return set()
        smallest_r = sorted(self.cache)[:min(r, len(self.cache))]
        return set(smallest_r)

    # ------------------------ 分值/受害者选择 ------------------------
    def _score(self, block_id: int) -> float:
        """
        越大越重要（不该被驱逐）：
          score = Wf * freq
                + Wr * recency
                + Wi * id_bias       (小编号更大)
        其中 recency = 1 / (1 + step - last_access)
            id_bias = 1 / block_id
        """
        f = float(self.freq.get(block_id, 0))
        la = self.last_access.get(block_id, 0)
        rec = 1.0 / (1.0 + (self.step - la))  # 越近越大
        id_bias = 1.0 / float(block_id) if block_id > 0 else 1.0
        return self.W_FREQ * f + self.W_RECENCY * rec + self.W_IDBIAS * id_bias

    def _pick_victim(self) -> int:
        """
        两阶段挑选牺牲者：
        1) 先在 非前缀集合 里选分值最低者；
        2) 若非前缀为空（全是前缀），选前缀里编号最大的（尽量不破坏小编号前缀）。
        """
        if not self.cache:
            raise RuntimeError("cache empty when picking victim")

        pinned = self._current_pinned_set()
        non_prefix = [bid for bid in self.cache if bid not in pinned]

        if non_prefix:
            # 在非前缀里选“最不重要”的
            victim = min(non_prefix, key=self._score)
            return victim
        else:
            # 万不得已：从前缀里选编号最大（对前缀破坏最小）
            victim = max(self.cache)
            return victim

    # ------------------------ 统计裁剪（控制额外内存） ------------------------
    def _maybe_gc_stats(self) -> None:
        limit = self.STATS_GC_MULTIPLIER * max(1, self.cache_size)
        if len(self.freq) <= limit and len(self.last_access) <= limit:
            return
        # 只保留“当前在 cache 里”或“最近/频繁”的少量条目
        keep: Set[int] = set(self.cache)
        # 补一些最高频/最近访问项
        if len(keep) < limit // 2:
            top_by_freq = sorted(self.freq.items(), key=lambda kv: kv[1], reverse=True)[: (limit // 2)]
            keep.update(k for k, _ in top_by_freq)
        if len(keep) < limit:
            top_by_rec = sorted(self.last_access.items(), key=lambda kv: kv[1], reverse=True)[: (limit - len(keep))]
            keep.update(k for k, _ in top_by_rec)

        # 清理
        for d in (self.freq, self.last_access):
            for k in list(d.keys()):
                if k not in keep:
                    d.pop(k, None)

    # ------------------------ 主入口 ------------------------
    def access(
        self,
        block_id: int,
        prompt_blocks: Iterable[int],
        meta: Dict[str, Any] | None = None,
    ) -> bool:
        # 步进时间 &（在“一个 prompt 开始”时）更新前缀 EMA
        self.step += 1
        try:
            self._maybe_update_prefix_ema(prompt_blocks)
        except Exception:
            # 避免解析 prompt_blocks 异常影响主逻辑
            pass

        # 命中：只更新统计
        if block_id in self.cache:
            self.freq[block_id] += 1
            self.last_access[block_id] = self.step
            return True

        # 未命中：如必要先驱逐
        if len(self.cache) >= self.cache_size:
            victim = self._pick_victim()
            self.cache.remove(victim)
            self._backend_del(victim)

        # 插入新块
        self.cache.add(block_id)
        self._backend_add(block_id)
        self.freq[block_id] += 1
        self.last_access[block_id] = self.step

        # 偶尔做一次统计裁剪，控制内存
        if (self.step & 0xFF) == 0:  # 每 256 次访问做一次
            self._maybe_gc_stats()

        return False
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
    total = 0
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
            total += len(prompt_blocks)
            for block_id in prompt_blocks:
                events.append((prompt_blocks, block_id, meta))
    print(f"已加载 trace 文件 {path}，共 {len(events)} 次访问，涉及 {total} 个 block。")
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
