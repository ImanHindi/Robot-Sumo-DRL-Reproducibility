from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

from experiment_utils import (
    as_relative_or_absolute,
    base_config,
    memory_snapshot,
    release_runner_memory,
    run_webots,
)


def csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def status_complete(agent_dir: Path) -> bool:
    model = agent_dir / "model_final.zip"
    if not model.exists() or model.stat().st_size <= 0:
        return False
    status = agent_dir / "status.json"
    if not status.exists():
        return True
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"
    except Exception:
        return True


def disk_guard(path: Path, minimum_free_gb: float) -> None:
    import psutil

    usage = psutil.disk_usage(str(path.resolve().anchor or path.resolve()))
    free_gb = usage.free / (1024.0 ** 3)
    print(f"[disk] free={free_gb:.1f} GB minimum={minimum_free_gb:.1f} GB")
    if free_gb < minimum_free_gb:
        raise OSError(
            f"Only {free_gb:.1f} GB free on the results drive. "
            f"At least {minimum_free_gb:.1f} GB is required before starting another Webots phase."
        )


def checkpoint_files(agent_dir: Path) -> list[Path]:
    folder = agent_dir / "checkpoints"
    found: list[tuple[int, Path]] = []
    for p in folder.glob("checkpoint_*_steps.zip"):
        m = re.search(r"checkpoint_(\d+)_steps\.zip$", p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def select_checkpoint_by_local_rank(agent_dir: Path, target_local_steps: int, checkpoint_every: int) -> Path:
    """Select synchronized checkpoint by its local Stage-3 rank.

    Veteran model filenames contain cumulative SB3 timesteps (pretraining + Stage 3),
    whereas the novice filenames start near zero.  The k-th checkpoint is nevertheless
    generated after the same number of Stage-3 environment steps for both controllers.
    """
    if target_local_steps <= 0 or target_local_steps % checkpoint_every != 0:
        raise ValueError("target_local_steps must be a positive multiple of checkpoint_every")
    files = checkpoint_files(agent_dir)
    rank = target_local_steps // checkpoint_every
    if len(files) < rank:
        raise FileNotFoundError(
            f"Need checkpoint rank {rank} ({target_local_steps} local steps) in {agent_dir}, "
            f"but only {len(files)} checkpoints were found."
        )
    return files[rank - 1]


def train_selfplay(
    pair_root: Path,
    veteran_model: Path,
    algorithm: str,
    seed: int,
    timesteps: int,
    checkpoint_every: int,
    webots: str | None,
    visible: bool,
    resume: bool,
    min_disk_gb: float,
) -> tuple[Path, Path]:
    run_dir = pair_root / "stage3_selfplay"
    a0 = run_dir / "agent_0"
    a1 = run_dir / "agent_1"
    veteran_final = a0 / "model_final.zip"
    novice_final = a1 / "model_final.zip"

    if resume and status_complete(a0) and status_complete(a1):
        print(f"[resume] SKIP completed Stage-3 self-play seed={seed}")
        return veteran_final, novice_final

    # A partially completed two-learning-agent phase cannot be resumed from only
    # one side without changing the opponent trajectory.  Restart THIS PAIR's
    # Stage-3 phase cleanly from the frozen veteran source + fresh novice.  Earlier
    # completed seeds remain untouched by the master resume runner.
    if run_dir.exists():
        print(f"[resume] incomplete Stage-3 pair detected for seed={seed}; restarting this pair cleanly")
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(pair_root / "head_to_head", ignore_errors=True)
        shutil.rmtree(pair_root / "external", ignore_errors=True)

    disk_guard(pair_root, min_disk_gb)
    cfg = base_config()
    cfg["experiment"].update(
        id=f"asym_selfplay_seed{seed}_train",
        stage="stage3_asymmetric_veteran_novice_selfplay",
        seed=int(seed),
        run_dir=as_relative_or_absolute(run_dir),
        notes=(
            "v1.14 add-on experiment over frozen v1.12 physics/reward. "
            "Agent 0 resumes the 500k-step PPO veteran; Agent 1 starts from random initialization."
        ),
    )
    cfg["training"]["total_timesteps"] = int(timesteps)
    cfg["training"]["checkpoint_every_steps"] = int(checkpoint_every)
    # Training episode CSVs are sufficient for publication trajectories; suppress
    # TensorBoard growth because the user's full benchmark previously exhausted disk.
    cfg["training"]["tensorboard"] = False
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="train", algorithm=algorithm, resume_model=str(veteran_model.resolve()))
    cfg["agents"]["1"].update(mode="train", algorithm=algorithm, resume_model=None)
    cfg["termination"]["required_done_agents"] = [0, 1]

    print(f"[selfplay] TRAIN seed={seed} algorithm={algorithm} stage3_steps={timesteps}")
    print(f"[selfplay] veteran={veteran_model}")
    run_webots(cfg, webots, visible)
    if not veteran_final.exists() or not novice_final.exists():
        raise FileNotFoundError("Stage-3 self-play did not produce both final models")
    return veteran_final, novice_final


def eval_head_to_head(
    pair_root: Path,
    label: str,
    model0: Path,
    model1: Path,
    algorithm: str,
    episodes: int,
    eval_seed: int,
    webots: str | None,
    visible: bool,
    resume: bool,
    min_disk_gb: float,
) -> Path:
    run_dir = pair_root / "head_to_head" / label
    out = run_dir / "agent_0" / "evaluation.csv"
    if resume and csv_rows(out) >= episodes:
        print(f"[resume] SKIP H2H {label}: {csv_rows(out)}/{episodes}")
        return out

    disk_guard(pair_root, min_disk_gb)
    cfg = base_config()
    cfg["experiment"].update(
        id=f"{pair_root.name}_{label}",
        stage=f"selfplay_h2h_{label}",
        seed=int(eval_seed),
        run_dir=as_relative_or_absolute(run_dir),
    )
    cfg["evaluation"]["episodes"] = int(episodes)
    cfg["evaluation"]["deterministic"] = True
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=algorithm, model_path=str(model0.resolve()))
    cfg["agents"]["1"].update(
        mode="frozen",
        algorithm=algorithm,
        model_path=str(model1.resolve()),
        deterministic_eval=True,
    )
    cfg["termination"]["required_done_agents"] = [0]
    print(f"[selfplay] H2H {label} episodes={episodes} seed={eval_seed}")
    run_webots(cfg, webots, visible)
    return out


def eval_external(
    pair_root: Path,
    actor_label: str,
    model: Path,
    algorithm: str,
    opponent_label: str,
    episodes: int,
    eval_seed: int,
    webots: str | None,
    visible: bool,
    resume: bool,
    min_disk_gb: float,
) -> Path:
    run_dir = pair_root / "external" / actor_label / opponent_label
    out = run_dir / "agent_0" / "evaluation.csv"
    if resume and csv_rows(out) >= episodes:
        print(f"[resume] SKIP external {actor_label} vs {opponent_label}: {csv_rows(out)}/{episodes}")
        return out

    disk_guard(pair_root, min_disk_gb)
    cfg = base_config()
    cfg["experiment"].update(
        id=f"{pair_root.name}_{actor_label}_{opponent_label}",
        stage=f"selfplay_external_{actor_label}_{opponent_label}",
        seed=int(eval_seed),
        run_dir=as_relative_or_absolute(run_dir),
    )
    cfg["evaluation"]["episodes"] = int(episodes)
    cfg["evaluation"]["deterministic"] = True
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=algorithm, model_path=str(model.resolve()))
    if opponent_label == "static":
        cfg["agents"]["1"].update(mode="static")
    elif opponent_label == "scripted":
        cfg["agents"]["1"].update(mode="scripted", scripted_profile="standard")
    elif opponent_label == "aggressive":
        cfg["agents"]["1"].update(mode="scripted", scripted_profile="aggressive")
    else:
        raise ValueError(opponent_label)
    cfg["termination"]["required_done_agents"] = [0]
    print(f"[selfplay] EXTERNAL {actor_label} vs {opponent_label} episodes={episodes}")
    run_webots(cfg, webots, visible)
    return out


def prune_unused_checkpoints(agent_dir: Path, keep: Iterable[Path]) -> None:
    keep_resolved = {p.resolve() for p in keep}
    removed = 0
    for p in checkpoint_files(agent_dir):
        if p.resolve() not in keep_resolved:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[disk] pruned {removed} unused checkpoints from {agent_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="One paired asymmetric veteran-vs-novice PPO self-play replicate")
    p.add_argument("--benchmark-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--algorithm", default="PPO", choices=["PPO"])
    p.add_argument("--stage3-steps", type=int, default=500000)
    p.add_argument("--checkpoint-every", type=int, default=50000)
    p.add_argument("--checkpoint-eval-steps", nargs="+", type=int, default=[100000, 200000, 300000, 400000, 500000])
    p.add_argument("--intermediate-episodes", type=int, default=50)
    p.add_argument("--final-h2h-episodes", type=int, default=100)
    p.add_argument("--external-episodes", type=int, default=100)
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--min-free-disk-gb", type=float, default=float(os.environ.get("SUMO_RL_MIN_FREE_DISK_GB", "20")))
    args = p.parse_args()

    benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pair_root = output_root / "pairs" / f"seed_{args.seed}"
    pair_root.mkdir(parents=True, exist_ok=True)
    veteran_model = benchmark_root / args.algorithm / f"seed_{args.seed}" / "stage2_scripted" / "agent_0" / "model_final.zip"
    if not veteran_model.exists():
        raise FileNotFoundError(f"Veteran model not found: {veteran_model}")

    manifest = {
        "experiment": "asymmetric_veteran_novice_selfplay",
        "runner_version": "v1.14",
        "frozen_environment": "v1.12",
        "algorithm": args.algorithm,
        "pair_seed": args.seed,
        "veteran_source_model": str(veteran_model),
        "veteran_pretraining_steps": 500000,
        "novice_initialization": "fresh_random_PPO",
        "stage3_selfplay_steps": args.stage3_steps,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_eval_steps": list(args.checkpoint_eval_steps),
        "intermediate_h2h_episodes_per_slot": args.intermediate_episodes,
        "final_h2h_episodes_per_slot": args.final_h2h_episodes,
        "external_episodes_per_condition": args.external_episodes,
        "deterministic_evaluation": True,
        "role_swap": True,
    }
    (pair_root / "PAIR_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    resume = not args.no_resume
    veteran_final, novice_final = train_selfplay(
        pair_root, veteran_model, args.algorithm, args.seed, args.stage3_steps,
        args.checkpoint_every, args.webots, args.visible, resume, args.min_free_disk_gb,
    )
    release_runner_memory(); memory_snapshot(f"after self-play training seed={args.seed}")

    train_dir = pair_root / "stage3_selfplay"
    veteran_agent_dir = train_dir / "agent_0"
    novice_agent_dir = train_dir / "agent_1"
    keep_v: list[Path] = []
    keep_n: list[Path] = []

    checkpoints = sorted(set(int(x) for x in args.checkpoint_eval_steps if 0 < int(x) <= args.stage3_steps))
    if args.stage3_steps not in checkpoints:
        checkpoints.append(args.stage3_steps)

    for local_steps in checkpoints:
        if local_steps == args.stage3_steps:
            vet_model = veteran_final
            nov_model = novice_final
            episodes = args.final_h2h_episodes
        else:
            vet_model = select_checkpoint_by_local_rank(veteran_agent_dir, local_steps, args.checkpoint_every)
            nov_model = select_checkpoint_by_local_rank(novice_agent_dir, local_steps, args.checkpoint_every)
            keep_v.append(vet_model); keep_n.append(nov_model)
            episodes = args.intermediate_episodes

        # Same spawn seed for both role orders to reduce slot/spawn noise.
        eval_seed = 500000 + args.seed * 10000 + int(local_steps // 1000)
        step_label = f"step_{local_steps}"
        eval_head_to_head(
            pair_root, f"{step_label}_veteran_as_agent0", vet_model, nov_model,
            args.algorithm, episodes, eval_seed, args.webots, args.visible, resume, args.min_free_disk_gb,
        )
        release_runner_memory()
        eval_head_to_head(
            pair_root, f"{step_label}_novice_as_agent0", nov_model, vet_model,
            args.algorithm, episodes, eval_seed, args.webots, args.visible, resume, args.min_free_disk_gb,
        )
        release_runner_memory()

    # External generalization after self-play.  Use the exact v1.12 benchmark
    # evaluation seed so veteran-pre and post-self-play policies face the same
    # deterministic spawn sequence for paired interpretation.
    external_eval_seed = 100000 + args.seed
    for actor_label, model in (("veteran_post", veteran_final), ("novice_post", novice_final)):
        for opp in ("static", "scripted", "aggressive"):
            eval_external(
                pair_root, actor_label, model, args.algorithm, opp, args.external_episodes,
                external_eval_seed, args.webots, args.visible, resume, args.min_free_disk_gb,
            )
            release_runner_memory()

    if not args.no_prune:
        prune_unused_checkpoints(veteran_agent_dir, keep_v)
        prune_unused_checkpoints(novice_agent_dir, keep_n)

    (pair_root / "PAIR_COMPLETE.txt").write_text(
        f"complete seed={args.seed} time={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    print(f"[selfplay] pair complete: {pair_root}")


if __name__ == "__main__":
    main()
