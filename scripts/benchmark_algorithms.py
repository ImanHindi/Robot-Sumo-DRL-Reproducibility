from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from experiment_utils import (
    ROOT,
    as_relative_or_absolute,
    base_config,
    memory_snapshot,
    release_runner_memory,
    results_root,
    run_webots,
    timestamp_id,
)


def _evaluation_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def _stage_complete(run_dir: Path) -> bool:
    model = run_dir / "agent_0" / "model_final.zip"
    if not model.exists() or model.stat().st_size <= 0:
        return False
    status = run_dir / "agent_0" / "status.json"
    if not status.exists():
        return True  # model file is the authoritative artifact
    try:
        payload = json.loads(status.read_text(encoding="utf-8"))
        return payload.get("status") == "complete"
    except Exception:
        return True


def train_phase(
    root: Path,
    algorithm: str,
    seed: int,
    stage: str,
    opponent: str,
    timesteps: int,
    resume: str | None,
    webots: str | None,
    visible: bool,
    profile: str = "standard",
    curriculum_profile: str | None = None,
    resume_existing: bool = True,
) -> Path:
    run_dir = root / algorithm / f"seed_{seed}" / stage
    model = run_dir / "agent_0" / "model_final.zip"
    if resume_existing and _stage_complete(run_dir):
        print(f"[resume] SKIP completed training: {algorithm} seed={seed} {stage}")
        print(f"[resume] model={model}")
        return model

    cfg = base_config()
    cfg["experiment"].update(
        id=f"{root.name}_{algorithm}_seed{seed}_{stage}",
        stage=stage,
        seed=seed,
        run_dir=as_relative_or_absolute(run_dir),
    )
    cfg["training"]["total_timesteps"] = int(timesteps)
    if curriculum_profile:
        cfg.setdefault("curriculum", {})["profile"] = curriculum_profile
    cfg["agents"]["0"].update(mode="train", algorithm=algorithm, resume_model=resume)
    cfg["agents"]["1"].update(mode=opponent, scripted_profile=profile)
    cfg["termination"]["required_done_agents"] = [0]

    print(f"[run] TRAIN {algorithm} seed={seed} stage={stage} steps={timesteps}")
    run_webots(cfg, webots, visible)
    if not model.exists():
        raise FileNotFoundError(f"Expected model not produced: {model}")
    return model


def eval_phase(
    root: Path,
    algorithm: str,
    seed: int,
    model: Path,
    label: str,
    opponent: str,
    episodes: int,
    webots: str | None,
    visible: bool,
    profile: str = "standard",
    resume_existing: bool = True,
) -> Path:
    run_dir = root / algorithm / f"seed_{seed}" / f"eval_{label}"
    out_csv = run_dir / "agent_0" / "evaluation.csv"
    existing_rows = _evaluation_rows(out_csv)
    if resume_existing and existing_rows >= int(episodes):
        print(
            f"[resume] SKIP completed evaluation: {algorithm} seed={seed} "
            f"{label} rows={existing_rows}/{episodes}"
        )
        return out_csv

    if existing_rows:
        print(
            f"[resume] partial evaluation found ({existing_rows}/{episodes}) for "
            f"{algorithm} seed={seed} {label}; rerunning this evaluation condition cleanly"
        )

    cfg = base_config()
    cfg["experiment"].update(
        id=f"{root.name}_{algorithm}_seed{seed}_eval_{label}",
        stage=f"eval_{label}",
        seed=100000 + seed,
        run_dir=as_relative_or_absolute(run_dir),
    )
    cfg["evaluation"]["episodes"] = int(episodes)
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=algorithm, model_path=str(model.resolve()))
    cfg["agents"]["1"].update(mode=opponent, scripted_profile=profile)
    cfg["termination"]["required_done_agents"] = [0]

    print(f"[run] EVAL {algorithm} seed={seed} vs={label} episodes={episodes}")
    run_webots(cfg, webots, visible)
    return out_csv


def analyze(root: Path) -> None:
    analyzer = ROOT / "analysis" / "analyze_benchmark.py"
    subprocess.run([sys.executable, str(analyzer), "--root", str(root)], check=True)


def main():
    p = argparse.ArgumentParser(description="Publication-oriented benchmark across RL algorithms")
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
    p.add_argument("--skip-stage1", action="store_true", help="Train directly against moving opponent")
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--name", default="benchmark")
    p.add_argument("--results-root", default=None, help="Short output root; Windows default is ~/SumoRL_results")
    p.add_argument(
        "--resume-root",
        default=None,
        help=(
            "Existing benchmark directory to continue. Completed training phases and "
            "evaluation conditions are skipped. Example: "
            r"C:\Users\Name\SumoRL_results\20260914_010203_benchmark_v112"
        ),
    )
    p.add_argument(
        "--no-resume-existing",
        action="store_true",
        help="Do not skip existing completed phase artifacts inside --resume-root.",
    )
    p.add_argument("--no-analyze", action="store_true", help="Skip central analysis at the end.")
    p.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.0,
        help="Small pause after each Webots phase so Windows can reclaim process/graphics memory.",
    )
    args = p.parse_args()

    if args.resume_root:
        root = Path(args.resume_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--resume-root does not exist: {root}")
        print(f"[resume] continuing benchmark root: {root}")
    else:
        benchmark_id = timestamp_id(args.name)
        root = results_root(args.results_root) / benchmark_id
        root.mkdir(parents=True, exist_ok=True)
        print(f"[benchmark] new root: {root}")

    resume_existing = not args.no_resume_existing
    memory_snapshot("benchmark worker start")

    for algo in args.algorithms:
        for seed in args.seeds:
            print(f"\n=== {algo} seed={seed} ===")
            final_stage_dir = root / algo / f"seed_{seed}" / "stage2_scripted"
            final_model_path = final_stage_dir / "agent_0" / "model_final.zip"

            # If Stage 2 is already complete, Stage 1 is irrelevant to resumption.
            if resume_existing and _stage_complete(final_stage_dir):
                final_model = final_model_path
                print(f"[resume] final training already complete for {algo} seed={seed}")
            else:
                if args.skip_stage1:
                    stage1_model = None
                else:
                    stage1_model = train_phase(
                        root,
                        algo,
                        seed,
                        "stage1_static",
                        "static",
                        args.stage1_steps,
                        None,
                        args.webots,
                        args.visible,
                        curriculum_profile="stage1_progressive_static",
                        resume_existing=resume_existing,
                    )
                    release_runner_memory()
                    memory_snapshot(f"after {algo} seed={seed} stage1")
                    if args.cooldown_seconds > 0:
                        time.sleep(args.cooldown_seconds)

                final_model = train_phase(
                    root,
                    algo,
                    seed,
                    "stage2_scripted",
                    "scripted",
                    args.stage2_steps,
                    str(stage1_model.resolve()) if stage1_model else None,
                    args.webots,
                    args.visible,
                    "standard",
                    curriculum_profile="stage2_progressive_scripted",
                    resume_existing=resume_existing,
                )
                release_runner_memory()
                memory_snapshot(f"after {algo} seed={seed} stage2")
                if args.cooldown_seconds > 0:
                    time.sleep(args.cooldown_seconds)

            eval_phase(
                root,
                algo,
                seed,
                final_model,
                "static",
                "static",
                args.eval_episodes,
                args.webots,
                args.visible,
                resume_existing=resume_existing,
            )
            release_runner_memory()
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

            eval_phase(
                root,
                algo,
                seed,
                final_model,
                "scripted",
                "scripted",
                args.eval_episodes,
                args.webots,
                args.visible,
                "standard",
                resume_existing=resume_existing,
            )
            release_runner_memory()
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

            eval_phase(
                root,
                algo,
                seed,
                final_model,
                "aggressive",
                "scripted",
                args.eval_episodes,
                args.webots,
                args.visible,
                "aggressive",
                resume_existing=resume_existing,
            )
            release_runner_memory()
            memory_snapshot(f"completed {algo} seed={seed}")
            if args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

    if not args.no_analyze:
        analyze(root)
        print(f"\nBenchmark complete: {root}")
    else:
        print(f"\nBenchmark worker complete: {root}")


if __name__ == "__main__":
    main()
