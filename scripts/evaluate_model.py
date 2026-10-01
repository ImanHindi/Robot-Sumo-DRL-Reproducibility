from __future__ import annotations

import argparse
from pathlib import Path

from experiment_utils import as_relative_or_absolute, base_config, results_root, run_webots, timestamp_id


def main():
    p = argparse.ArgumentParser(description="Evaluate a frozen RL policy in fresh Webots matches")
    p.add_argument("--model", required=True)
    p.add_argument("--algorithm", required=True, choices=["A2C", "PPO", "DQN", "RecurrentPPO"])
    p.add_argument("--opponent", default="scripted", choices=["static", "scripted", "frozen"])
    p.add_argument("--opponent-model", default=None)
    p.add_argument("--opponent-algorithm", default="PPO")
    p.add_argument("--profile", default="standard", choices=["standard", "aggressive"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--name", default="evaluation")
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    p.add_argument("--results-root", default=None, help="Short output root; Windows default is ~/SumoRL_results")
    args = p.parse_args()

    run_id = timestamp_id(args.name)
    run_dir = results_root(args.results_root) / run_id
    cfg = base_config()
    cfg["experiment"].update(id=run_id, stage="evaluation", seed=args.seed, run_dir=as_relative_or_absolute(run_dir))
    cfg["evaluation"]["episodes"] = args.episodes
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=args.algorithm, model_path=str(Path(args.model).resolve()))
    cfg["agents"]["1"].update(mode=args.opponent, scripted_profile=args.profile)
    if args.opponent == "frozen":
        if not args.opponent_model:
            p.error("--opponent-model is required when --opponent frozen")
        cfg["agents"]["1"].update(algorithm=args.opponent_algorithm, model_path=str(Path(args.opponent_model).resolve()))
    cfg["termination"]["required_done_agents"] = [0]
    run_webots(cfg, args.webots, args.visible)
    print(f"Evaluation results: {run_dir / 'agent_0' / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
