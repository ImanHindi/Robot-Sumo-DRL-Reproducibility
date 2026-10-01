from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import as_relative_or_absolute, base_config, results_root, run_webots, timestamp_id


def main():
    p = argparse.ArgumentParser(description="Run one controlled Sumo RL training experiment")
    p.add_argument("--algorithm", default="PPO", choices=["A2C", "PPO", "DQN", "RecurrentPPO"])
    p.add_argument("--opponent", default="scripted", choices=["static", "scripted"])
    p.add_argument("--profile", default="standard", choices=["standard", "aggressive"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=500000)
    p.add_argument("--resume", default=None)
    p.add_argument("--name", default="single")
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--results-root", default=None, help="Short output root; Windows default is ~/SumoRL_results")
    args = p.parse_args()

    run_id = timestamp_id(args.name)
    run_dir = results_root(args.results_root) / run_id
    cfg = base_config()
    cfg["experiment"].update(id=run_id, stage=f"train_vs_{args.opponent}", seed=args.seed, run_dir=as_relative_or_absolute(run_dir))
    cfg["training"]["total_timesteps"] = args.timesteps
    cfg.setdefault("curriculum", {})["profile"] = (
        "stage1_progressive_static" if args.opponent == "static" else "stage2_progressive_scripted"
    )
    cfg["agents"]["0"].update(mode="train", algorithm=args.algorithm, resume_model=args.resume)
    cfg["agents"]["1"].update(mode=args.opponent, scripted_profile=args.profile)
    cfg["termination"]["required_done_agents"] = [0]
    run_webots(cfg, args.webots, args.visible)
    print(f"Completed. Results: {run_dir}")


if __name__ == "__main__":
    main()
