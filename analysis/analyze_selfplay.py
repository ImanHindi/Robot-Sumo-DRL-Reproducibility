from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / "paper_results"
    out.mkdir(parents=True, exist_ok=True)

    labels = {
        "eval_veteran_pre": "Veteran pre-self-play",
        "eval_veteran_post": "Veteran post-self-play",
        "eval_novice_post": "Novice post-self-play",
        "head_to_head_veteran_as_agent0": "Veteran vs novice",
        "head_to_head_novice_as_agent0": "Novice vs veteran",
    }
    frames = []
    rows = []
    for folder, label in labels.items():
        path = root / folder / "agent_0" / "evaluation.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["condition_label"] = label
        frames.append(df)
        rows.append({
            "condition": label,
            "episodes": len(df),
            "win_rate": df.win.mean(),
            "loss_rate": df.loss.mean(),
            "draw_rate": df.draw.mean(),
            "forced_win_rate": df.forced_win.mean() if "forced_win" in df.columns else 0.0,
            "unforced_win_rate": df.unforced_win.mean() if "unforced_win" in df.columns else 0.0,
            "mean_reward": df.episode_reward.mean(),
            "median_steps": df.episode_steps.median(),
            "aggression_index": df.aggression_index.mean(),
            "engagement_ratio": df.engagement_ratio.mean(),
            "edge_risk_ratio": df.edge_risk_ratio.mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "selfplay_summary.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "selfplay_all_episodes.csv", index=False)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        ax.bar(summary.condition, summary.win_rate)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Win rate for evaluated Agent 0")
        ax.set_title("Asymmetric self-play outcomes")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out / "selfplay_win_rates.png", dpi=300)
        fig.savefig(out / "selfplay_win_rates.pdf")
        plt.close(fig)

        base = summary[summary.condition.isin(["Veteran pre-self-play", "Veteran post-self-play", "Novice post-self-play"])]
        if not base.empty:
            fig, ax = plt.subplots(figsize=(7.6, 4.6))
            ax.bar(base.condition, base.aggression_index)
            ax.set_ylabel("Aggression index")
            ax.set_title("Behavioral change before and after asymmetric self-play")
            ax.tick_params(axis="x", rotation=15)
            fig.tight_layout()
            fig.savefig(out / "selfplay_aggression_change.png", dpi=300)
            fig.savefig(out / "selfplay_aggression_change.pdf")
            plt.close(fig)
    print(f"Self-play analysis written to {out}")


if __name__ == "__main__":
    main()
