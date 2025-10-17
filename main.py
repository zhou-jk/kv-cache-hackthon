from __future__ import annotations

import argparse
from pathlib import Path

from policies import compare_policies, create_policy, load_trace_from_jsonl, run_policy


def build_sample_events():
    sample = [
        ([1, 2, 3, 4], {"input_length": 4, "round": 0, "type": 1}),
        ([1, 2, 3, 4, 5], {"input_length": 5, "round": 1, "type": 1}),
        ([1, 2, 6], {"input_length": 3, "round": 1, "type": 2}),
    ]
    events = []
    for blocks, meta in sample:
        for block in blocks:
            events.append((blocks, block, meta))
    return events


def parse_args() -> argparse.Namespace:
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
        "--prefix-keep",
        type=int,
        default=64,
        help="增强策略：优先保留 prompt 前若干 block",
    )
    parser.add_argument(
        "--block-field",
        default="hash_ids",
        help="JSONL 中存放 block 列表的字段名",
    )
    parser.add_argument(
        "--policies",
        default="advanced",
        help="要运行的策略列表，逗号分隔（如：fifo,lru,prefix,advanced）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.trace:
        events = load_trace_from_jsonl(args.trace, block_field=args.block_field)
    else:
        events = build_sample_events()

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
