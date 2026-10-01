from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import ROOT, memory_snapshot, results_root, timestamp_id


def _wait_for_recovery(min_free_mb: float, timeout_s: float) -> None:
    import psutil

    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        free_mb = psutil.virtual_memory().available / (1024.0 ** 2)
        if free_mb >= min_free_mb:
            return
        if time.monotonic() >= deadline:
            raise MemoryError(
                f"System memory did not recover to {min_free_mb:.0f} MB available "
                f"within {timeout_s:.0f}s. Close other applications, then resume the "
                "same benchmark root; completed phases will be skipped."
            )
        print(
            f"[memory] waiting for Windows to reclaim RAM: "
            f"available={free_mb:.0f} MB target={min_free_mb:.0f} MB"
        )
        time.sleep(5.0)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Memory-safe master benchmark. Runs each algorithm/seed in a fresh "
            "Python worker and resumes completed v1.12 phases after interruption."
        )
    )
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["A2C", "PPO", "DQN", "RecurrentPPO"],
        choices=["A2C", "PPO", "DQN", "RecurrentPPO"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--stage1-steps", type=int, default=200000)
    p.add_argument("--stage2-steps", type=int, default=300000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--name", default="benchmark_v112_memorysafe")
    p.add_argument("--results-root", default=None)
    p.add_argument(
        "--resume-root",
        default=None,
        help="Continue an existing v1.12/v1.13 benchmark directory instead of creating a new one.",
    )
    p.add_argument("--cooldown-seconds", type=float, default=3.0)
    p.add_argument(
        "--between-worker-min-free-mb",
        type=float,
        default=float(os.environ.get("SUMO_RL_BETWEEN_WORKER_MIN_FREE_MB", "1024")),
    )
    p.add_argument("--memory-recovery-timeout", type=float, default=90.0)
    args = p.parse_args()

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--resume-root does not exist: {root}")
        print(f"[memory-safe] RESUME ROOT: {root}")
    else:
        root = results_root(args.results_root) / timestamp_id(args.name)
        root.mkdir(parents=True, exist_ok=True)
        print(f"[memory-safe] NEW ROOT: {root}")

    # Persist the exact command users can copy after any Windows/OOM interruption.
    resume_txt = root / "MEMORY_SAFE_RESUME.txt"
    algo_text = " ".join(args.algorithms)
    seed_text = " ".join(str(seed) for seed in args.seeds)
    master_script = ROOT / "scripts" / "benchmark_memory_safe.py"
    resume_cmd = (
        f'python "{master_script}" '
        f'--resume-root "{root}" --algorithms {algo_text} '
        f'--seeds {seed_text} '
        f'--stage1-steps {args.stage1_steps} --stage2-steps {args.stage2_steps} '
        f'--eval-episodes {args.eval_episodes}'
    )
    resume_txt.write_text(
        "Memory-safe benchmark resume command\n"
        "====================================\n"
        "Run from the activated sumo_rl_research environment.\n\n"
        + resume_cmd
        + "\n",
        encoding="utf-8",
    )

    worker = ROOT / "scripts" / "benchmark_algorithms.py"
    total = len(args.algorithms) * len(args.seeds)
    index = 0

    memory_snapshot("memory-safe master start")
    for algo in args.algorithms:
        for seed in args.seeds:
            index += 1
            print("\n" + "=" * 78)
            print(f"[memory-safe] WORKER {index}/{total}: {algo} seed={seed}")
            print(f"[memory-safe] root={root}")
            print("=" * 78)

            _wait_for_recovery(args.between_worker_min_free_mb, args.memory_recovery_timeout)

            cmd = [
                sys.executable,
                str(worker),
                "--algorithms",
                algo,
                "--seeds",
                str(seed),
                "--stage1-steps",
                str(args.stage1_steps),
                "--stage2-steps",
                str(args.stage2_steps),
                "--eval-episodes",
                str(args.eval_episodes),
                "--resume-root",
                str(root),
                "--no-analyze",
                "--cooldown-seconds",
                str(args.cooldown_seconds),
            ]
            if args.webots:
                cmd.extend(["--webots", args.webots])
            if args.visible:
                cmd.append("--visible")

            completed = subprocess.run(cmd, cwd=str(ROOT))
            if completed.returncode != 0:
                print("\n[memory-safe] Worker stopped before benchmark completion.")
                print("[memory-safe] Completed phases are preserved and will be skipped on resume.")
                print(f"[memory-safe] Resume root: {root}")
                print(f"[memory-safe] Saved command: {resume_txt}")
                raise SystemExit(completed.returncode)

            gc.collect()
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)
            memory_snapshot(f"after worker {index}/{total} {algo} seed={seed}")

    print("\n[memory-safe] All training/evaluation workers complete. Running central analysis...")
    analyzer = ROOT / "analysis" / "analyze_benchmark.py"
    subprocess.run([sys.executable, str(analyzer), "--root", str(root)], check=True)
    memory_snapshot("after central analysis")
    print(f"\nBenchmark complete: {root}")


if __name__ == "__main__":
    main()
