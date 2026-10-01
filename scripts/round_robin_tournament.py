from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

from experiment_utils import ROOT, as_relative_or_absolute, base_config, run_webots

ALGOS = ["A2C", "PPO", "DQN", "RecurrentPPO"]


def run_match(root: Path, seed: int, eval_algo: str, eval_model: Path,
              opp_algo: str, opp_model: Path, label: str, episodes: int,
              webots: str | None, visible: bool):
    run_dir = root / "tournament" / f"seed_{seed}" / label
    cfg = base_config()
    cfg["experiment"].update(
        id=f"{root.name}_tournament_seed{seed}_{label}",
        stage="round_robin_tournament",
        seed=200000 + seed,
        run_dir=as_relative_or_absolute(run_dir),
    )
    cfg["evaluation"]["episodes"] = episodes
    cfg.setdefault("curriculum", {})["profile"] = "evaluation_randomized"
    cfg["agents"]["0"].update(mode="eval", algorithm=eval_algo, model_path=str(eval_model.resolve()))
    cfg["agents"]["1"].update(mode="frozen", algorithm=opp_algo, model_path=str(opp_model.resolve()))
    cfg["termination"]["required_done_agents"] = [0]
    run_webots(cfg, webots, visible)


def main():
    p = argparse.ArgumentParser(description="Pairwise tournament among trained benchmark policies")
    p.add_argument("--root", required=True, help="Existing benchmark results root")
    p.add_argument("--algorithms", nargs="+", default=ALGOS, choices=ALGOS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--webots", default=None)
    p.add_argument("--visible", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()
    for seed in args.seeds:
        models = {}
        for algo in args.algorithms:
            model = root / algo / f"seed_{seed}" / "stage2_scripted" / "agent_0" / "model_final.zip"
            if model.exists():
                models[algo] = model
        for a, b in itertools.combinations(sorted(models), 2):
            print(f"\nTournament seed={seed}: {a} vs {b}")
            run_match(root, seed, a, models[a], b, models[b], f"{a}_as_agent0_vs_{b}", args.episodes, args.webots, args.visible)
            run_match(root, seed, b, models[b], a, models[a], f"{b}_as_agent0_vs_{a}", args.episodes, args.webots, args.visible)

    analyzer = ROOT / "analysis" / "analyze_tournament.py"
    subprocess.run([sys.executable, str(analyzer), "--root", str(root)], check=True)


if __name__ == "__main__":
    main()
