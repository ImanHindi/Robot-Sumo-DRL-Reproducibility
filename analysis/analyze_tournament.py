from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAT = re.compile(r"^(A2C|PPO|DQN|RecurrentPPO)_as_agent0_vs_(A2C|PPO|DQN|RecurrentPPO)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    tournament = root / "tournament"
    rows = []
    for eval_csv in tournament.rglob("evaluation.csv"):
        label = eval_csv.parents[1].name
        m = PAT.match(label)
        if not m:
            continue
        algo0, algo1 = m.groups()
        seed_name = next((x for x in eval_csv.parts if x.startswith("seed_")), "seed_-1")
        seed = int(seed_name.split("_")[-1])
        df = pd.read_csv(eval_csv)
        rows.append({
            "seed": seed,
            "agent0_algorithm": algo0,
            "agent1_algorithm": algo1,
            "matches": len(df),
            "agent0_win_rate": df.win.mean(),
            "agent0_loss_rate": df.loss.mean(),
            "draw_rate": df.draw.mean(),
            "agent0_forced_win_rate": df.forced_win.mean() if "forced_win" in df.columns else 0.0,
            "agent0_unforced_win_rate": df.unforced_win.mean() if "unforced_win" in df.columns else 0.0,
            "agent0_aggression": df.aggression_index.mean(),
        })
    if not rows:
        print("No tournament evaluations found")
        return
    raw = pd.DataFrame(rows)
    out = root / "paper_results" / "tournament"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "tournament_per_seed_side.csv", index=False)

    algos = sorted(set(raw.agent0_algorithm) | set(raw.agent1_algorithm))
    scores = {a: {b: [] for b in algos} for a in algos}
    for _, r in raw.iterrows():
        scores[r.agent0_algorithm][r.agent1_algorithm].append(r.agent0_win_rate)
        scores[r.agent1_algorithm][r.agent0_algorithm].append(r.agent0_loss_rate)
    matrix = pd.DataFrame(index=algos, columns=algos, dtype=float)
    for a in algos:
        for b in algos:
            if a == b:
                matrix.loc[a, b] = np.nan
            else:
                vals = scores[a][b]
                matrix.loc[a, b] = np.mean(vals) if vals else np.nan
    matrix.to_csv(out / "tournament_win_rate_matrix.csv")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(algos)), algos, rotation=30, ha="right")
    ax.set_yticks(range(len(algos)), algos)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Evaluated policy")
    ax.set_title("Round-robin win-rate matrix")
    for i in range(len(algos)):
        for j in range(len(algos)):
            v = matrix.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(out / "tournament_win_rate_matrix.png", dpi=300)
    fig.savefig(out / "tournament_win_rate_matrix.pdf")
    plt.close(fig)
    print(f"Tournament analysis written to {out}")


if __name__ == "__main__":
    main()
