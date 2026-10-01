from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_utils import ROOT, as_relative_or_absolute, base_config, results_root, run_webots, timestamp_id


def eval_one(root, label, model, algo, episodes, webots, visible, seed):
    run_dir = root / f"eval_{label}"
    cfg = base_config()
    cfg["experiment"].update(id=f"{root.name}_{label}", stage=f"eval_{label}", seed=seed, run_dir=as_relative_or_absolute(run_dir))
    cfg["evaluation"]["episodes"] = episodes
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=algo, model_path=str(Path(model).resolve()))
    cfg["agents"]["1"].update(mode="scripted", scripted_profile="standard")
    cfg["termination"]["required_done_agents"] = [0]
    run_webots(cfg, webots, visible)
    return run_dir / "agent_0" / "evaluation.csv"


def head_to_head(root, label, model0, algo0, model1, algo1, episodes, webots, visible, seed):
    run_dir = root / label
    cfg = base_config()
    cfg["experiment"].update(id=f"{root.name}_{label}", stage=label, seed=seed, run_dir=as_relative_or_absolute(run_dir))
    cfg["evaluation"]["episodes"] = episodes
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=algo0, model_path=str(Path(model0).resolve()))
    cfg["agents"]["1"].update(mode="frozen", algorithm=algo1, model_path=str(Path(model1).resolve()))
    cfg["termination"]["required_done_agents"] = [0]
    run_webots(cfg, webots, visible)
    return run_dir / "agent_0" / "evaluation.csv"


def main():
    p = argparse.ArgumentParser(description="Reproduce and quantify veteran-vs-novice asymmetric self-play")
    p.add_argument("--veteran-model", required=True)
    p.add_argument("--veteran-algorithm", default="PPO", choices=["A2C", "PPO", "DQN", "RecurrentPPO"])
    p.add_argument("--novice-algorithm", default=None, choices=["A2C", "PPO", "DQN", "RecurrentPPO"])
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--name", default="asymmetric_selfplay")
    p.add_argument("--results-root", default=None, help="Short output root; Windows default is ~/SumoRL_results")
    args = p.parse_args()
    novice_algo = args.novice_algorithm or args.veteran_algorithm

    root = results_root(args.results_root) / timestamp_id(args.name)
    root.mkdir(parents=True, exist_ok=True)

    # Baseline veteran evaluation before additional self-play.
    eval_one(root, "veteran_pre", args.veteran_model, args.veteran_algorithm,
             args.eval_episodes, args.webots, args.visible, args.seed + 1000)

    train_dir = root / "selfplay_training"
    cfg = base_config()
    cfg["experiment"].update(id=f"{root.name}_train", stage="stage3_asymmetric_selfplay", seed=args.seed, run_dir=as_relative_or_absolute(train_dir))
    cfg["training"]["total_timesteps"] = args.timesteps
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="train", algorithm=args.veteran_algorithm, resume_model=str(Path(args.veteran_model).resolve()))
    cfg["agents"]["1"].update(mode="train", algorithm=novice_algo, resume_model=None)
    cfg["termination"]["required_done_agents"] = [0, 1]
    run_webots(cfg, args.webots, args.visible)

    veteran_post = train_dir / "agent_0" / "model_final.zip"
    novice_post = train_dir / "agent_1" / "model_final.zip"
    if not veteran_post.exists() or not novice_post.exists():
        raise FileNotFoundError("Self-play did not produce both final models")

    eval_one(root, "veteran_post", veteran_post, args.veteran_algorithm,
             args.eval_episodes, args.webots, args.visible, args.seed + 2000)
    eval_one(root, "novice_post", novice_post, novice_algo,
             args.eval_episodes, args.webots, args.visible, args.seed + 3000)

    # Head-to-head in both robot slots to reduce side/spawn/controller bias.
    head_to_head(root, "head_to_head_veteran_as_agent0", veteran_post, args.veteran_algorithm,
                 novice_post, novice_algo, args.eval_episodes, args.webots, args.visible, args.seed + 4000)
    head_to_head(root, "head_to_head_novice_as_agent0", novice_post, novice_algo,
                 veteran_post, args.veteran_algorithm, args.eval_episodes, args.webots, args.visible, args.seed + 5000)

    analyzer = ROOT / "analysis" / "analyze_selfplay.py"
    subprocess.run([sys.executable, str(analyzer), "--root", str(root)], check=True)
    print(f"Self-play experiment complete: {root}")


if __name__ == "__main__":
    main()
