from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import ROOT, memory_snapshot, results_root, timestamp_id


def disk_free_gb(path: Path) -> float:
    import psutil
    anchor = Path(path.resolve().anchor or path.resolve())
    return psutil.disk_usage(str(anchor)).free / (1024.0 ** 3)


def memory_wait(min_free_mb: float, timeout_s: float) -> None:
    import psutil
    deadline = time.monotonic() + timeout_s
    while True:
        free = psutil.virtual_memory().available / (1024.0 ** 2)
        if free >= min_free_mb:
            return
        if time.monotonic() >= deadline:
            raise MemoryError(
                f"Available RAM did not recover to {min_free_mb:.0f} MB. "
                "Close other applications and resume the same self-play root."
            )
        print(f"[memory] waiting: available={free:.0f} MB target={min_free_mb:.0f} MB")
        time.sleep(5)


def resolve_benchmark_root(raw: str | None) -> Path:
    if raw:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Benchmark root does not exist: {p}")
        return p
    base = results_root(None)
    candidates = sorted(
        [p for p in base.glob("*_benchmark_v112") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if (p / "PPO" / "seed_0" / "stage2_scripted" / "agent_0" / "model_final.zip").exists():
            print(f"[auto] selected benchmark root: {p}")
            return p.resolve()
    raise FileNotFoundError(
        "Could not auto-discover a completed *_benchmark_v112 directory. "
        "Pass --benchmark-root explicitly."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Memory/disk-safe 5-pair asymmetric veteran-vs-novice PPO self-play study")
    p.add_argument("--benchmark-root", default=None)
    p.add_argument("--resume-root", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--stage3-steps", type=int, default=500000)
    p.add_argument("--checkpoint-every", type=int, default=50000)
    p.add_argument("--checkpoint-eval-steps", nargs="+", type=int, default=[100000, 200000, 300000, 400000, 500000])
    p.add_argument("--intermediate-episodes", type=int, default=50)
    p.add_argument("--final-h2h-episodes", type=int, default=100)
    p.add_argument("--external-episodes", type=int, default=100)
    p.add_argument("--results-root", default=None)
    p.add_argument("--name", default="asymmetric_selfplay_v114")
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--min-free-disk-gb", type=float, default=float(os.environ.get("SUMO_RL_MIN_FREE_DISK_GB", "20")))
    p.add_argument("--between-pair-min-free-mb", type=float, default=1024.0)
    p.add_argument("--memory-recovery-timeout", type=float, default=120.0)
    p.add_argument("--no-prune", action="store_true")
    args = p.parse_args()

    benchmark_root = resolve_benchmark_root(args.benchmark_root)
    # Validate all requested veteran models before investing compute.
    for seed in args.seeds:
        model = benchmark_root / "PPO" / f"seed_{seed}" / "stage2_scripted" / "agent_0" / "model_final.zip"
        if not model.exists():
            raise FileNotFoundError(f"Missing PPO veteran model for seed {seed}: {model}")

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Self-play resume root does not exist: {root}")
        print(f"[selfplay] RESUME ROOT: {root}")
    else:
        root = results_root(args.results_root) / timestamp_id(args.name)
        root.mkdir(parents=True, exist_ok=True)
        print(f"[selfplay] NEW ROOT: {root}")

    plan = {
        "runner": "v1.14_asymmetric_selfplay",
        "frozen_environment": "v1.12",
        "benchmark_root": str(benchmark_root),
        "algorithm": "PPO",
        "seeds": args.seeds,
        "veteran_pretraining_steps": 500000,
        "stage3_steps_per_pair": args.stage3_steps,
        "checkpoint_eval_steps": args.checkpoint_eval_steps,
        "role_swapped_deterministic_h2h": True,
        "external_generalization": ["static", "scripted", "aggressive"],
    }
    (root / "SELFPLAY_PLAN.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    resume_cmd = (
        f'python "{ROOT / "scripts" / "asymmetric_selfplay_master.py"}" '
        f'--benchmark-root "{benchmark_root}" --resume-root "{root}" '
        f'--seeds {" ".join(str(s) for s in args.seeds)} '
        f'--stage3-steps {args.stage3_steps} --checkpoint-every {args.checkpoint_every} '
        f'--checkpoint-eval-steps {" ".join(str(x) for x in args.checkpoint_eval_steps)} '
        f'--intermediate-episodes {args.intermediate_episodes} '
        f'--final-h2h-episodes {args.final_h2h_episodes} --external-episodes {args.external_episodes}'
    )
    (root / "RESUME_ASYMMETRIC_SELFPLAY.txt").write_text(resume_cmd + "\n", encoding="utf-8")

    worker = ROOT / "scripts" / "asymmetric_selfplay_pair.py"
    analyzer = ROOT / "analysis" / "analyze_asymmetric_selfplay.py"
    total = len(args.seeds)

    for i, seed in enumerate(args.seeds, 1):
        print("\n" + "=" * 80)
        print(f"[selfplay] PAIR {i}/{total}: PPO veteran seed {seed} vs fresh PPO novice")
        print("=" * 80)
        memory_wait(args.between_pair_min_free_mb, args.memory_recovery_timeout)
        free_gb = disk_free_gb(root)
        print(f"[disk] before pair seed={seed}: free={free_gb:.1f} GB")
        if free_gb < args.min_free_disk_gb:
            raise OSError(
                f"Only {free_gb:.1f} GB disk space remains. Free disk space, then resume using {root / 'RESUME_ASYMMETRIC_SELFPLAY.txt'}"
            )

        cmd = [
            sys.executable, str(worker),
            "--benchmark-root", str(benchmark_root),
            "--output-root", str(root),
            "--seed", str(seed),
            "--stage3-steps", str(args.stage3_steps),
            "--checkpoint-every", str(args.checkpoint_every),
            "--checkpoint-eval-steps", *[str(x) for x in args.checkpoint_eval_steps],
            "--intermediate-episodes", str(args.intermediate_episodes),
            "--final-h2h-episodes", str(args.final_h2h_episodes),
            "--external-episodes", str(args.external_episodes),
            "--min-free-disk-gb", str(args.min_free_disk_gb),
        ]
        if args.webots:
            cmd.extend(["--webots", args.webots])
        if args.visible:
            cmd.append("--visible")
        if args.no_prune:
            cmd.append("--no-prune")

        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            print(f"[selfplay] pair seed={seed} stopped. Completed phases are resumable.")
            print(f"[selfplay] resume instructions: {root / 'RESUME_ASYMMETRIC_SELFPLAY.txt'}")
            raise SystemExit(result.returncode)

        # Partial analysis is useful even if Windows is interrupted later.
        subprocess.run(
            [sys.executable, str(analyzer), "--root", str(root), "--benchmark-root", str(benchmark_root)],
            cwd=str(ROOT), check=True,
        )
        gc.collect()
        memory_snapshot(f"after self-play pair seed={seed}")
        time.sleep(3.0)

    subprocess.run(
        [sys.executable, str(analyzer), "--root", str(root), "--benchmark-root", str(benchmark_root)],
        cwd=str(ROOT), check=True,
    )
    print(f"\nAsymmetric self-play study complete: {root}")
    print(f"Paper results: {root / 'paper_results'}")


if __name__ == "__main__":
    main()
