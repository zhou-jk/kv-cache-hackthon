from __future__ import annotations

import argparse
from pathlib import Path

from policies import compare_policies, create_policy, load_trace_from_text, run_policy


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
        "--policies",
        default="advanced",
        help="要运行的策略列表，逗号分隔（如：fifo,lru,prefix,advanced）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_size = 0
    if args.trace:
        cache_size, events = load_trace_from_text(args.trace)
    else:
        events = build_sample_events()
    if cache_size != 0:
        args.cache_size = cache_size
    print(f"使用的缓存大小: {args.cache_size} blocks")

    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    if not policy_names:
        raise ValueError("必须至少指定一种策略")

    policies = [create_policy(name, args) for name in policy_names]

    compare_policies(policies, events)


if __name__ == "__main__":
    main()
