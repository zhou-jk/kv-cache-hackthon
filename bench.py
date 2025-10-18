from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt

from policies import compare_policies, create_policy, load_trace_from_text, run_policy


@dataclass
class PolicyArgs:
    cache_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量多线程 KV Cache 策略测试脚本")
    parser.add_argument(
        "--trace",
        nargs="+",
        required=True,
        help="要测试的 trace 文件路径（支持通配符）",
    )
    parser.add_argument(
        "--cache-sizes",
        required=False,
        help="逗号分隔的缓存大小列表，例如：128,256,512",
    )
    parser.add_argument(
        "--policies",
        required=True,
        help="要测试的策略列表，逗号分隔",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="结果保存的 CSV 路径",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("hit_rate_stability.png"),
        help="命中率稳定性图的输出路径",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="并行线程数",
    )
    parser.add_argument(
        "--cpu-affinity",
        help="将进程绑定到指定 CPU，上下文格式如 0,1,2",
    )
    return parser.parse_args()


def expand_trace_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matched = list(Path().glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
            else:
                raise FileNotFoundError(f"找不到 trace 文件：{pattern}")
    return paths


def run_single_case(
    trace_path: Path,
    policy_name: str,
    base_args: argparse.Namespace,
) -> dict:
    cache_size, events = load_trace_from_text(trace_path)
    policy_args = PolicyArgs(
        cache_size=cache_size,
    )
    policy = create_policy(policy_name, policy_args)
    misses, total = run_policy(policy, events)
    hits = total - misses
    hit_rate = hits / total if total else 0.0
    return {
        "trace": str(trace_path),
        "policy": policy.name,
        "cache_size": cache_size,
        "hits": hits,
        "misses": misses,
        "total": total,
        "hit_rate": hit_rate,
    }


def save_results_csv(output_csv: Path, rows: Sequence[dict]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_hit_rate_stability(output_plot: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    grouped: dict[str, dict[int, List[float]]] = {}
    for row in rows:
        policy = row["policy"]
        cache = row["cache_size"]
        hit_rate = row["hit_rate"]
        grouped.setdefault(policy, {}).setdefault(cache, []).append(hit_rate)

    cache_sizes = sorted({row["cache_size"] for row in rows})
    plt.figure(figsize=(10, 6))
    for policy, cache_map in grouped.items():
        y = []
        for cache in cache_sizes:
            values = cache_map.get(cache)
            y.append(mean(values) if values else 0.0)
        plt.plot(cache_sizes, y, marker="o", label=policy)

    plt.xlabel("Cache Size (blocks)")
    plt.ylabel("Hit Rate")
    plt.title("命中率稳定性")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=180)
    plt.close()


def set_cpu_affinity(affinity: str | None) -> None:
    if not affinity:
        return
    try:
        cpus = [int(x.strip()) for x in affinity.split(",") if x.strip()]
        if not cpus:
            return
        if sys.platform.startswith("win"):
            mask = 0
            for cpu in cpus:
                mask |= 1 << cpu
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
        elif hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        else:
            try:
                import psutil
                psutil.Process().cpu_affinity(cpus)
            except Exception as exc:
                print(f"警告: 无法设置 CPU 亲和性（需要 psutil）: {exc}")
    except Exception as exc:
        print(f"警告: 设置 CPU 亲和性失败: {exc}")


def main() -> None:
    args = parse_args()
    set_cpu_affinity(args.cpu_affinity)
    trace = expand_trace_paths(args.trace)
    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    base_args = argparse.Namespace(
        block_field="hash_ids",
    )

    tasks = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_case = {}
        for trace_path in trace:
            for policy_name in policy_names:
                future = executor.submit(
                    run_single_case,
                    trace_path,
                    policy_name,
                    base_args,
                )
                future_to_case[future] = (trace_path, policy_name)

        for future in as_completed(future_to_case):
            trace_path, policy_name = future_to_case[future]
            try:
                result = future.result()
                tasks.append(result)
                print(
                    f"[完成] trace={trace_path} "
                    f"policy={policy_name} hit_rate={result['hit_rate']:.2%}"
                )
            except Exception as exc:
                print(
                    f"[失败] trace={trace_path} "
                    f"policy={policy_name} error={exc}"
                )

    save_results_csv(args.output_csv, tasks)
    plot_hit_rate_stability(args.output_plot, tasks)
    print(f"结果已保存至 {args.output_csv}")
    print(f"命中率稳定性图已保存至 {args.output_plot}")


if __name__ == "__main__":
    main()
