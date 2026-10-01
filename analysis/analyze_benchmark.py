from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def read_evaluations(root: Path):
    frames = []
    for p in root.rglob("evaluation.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        if "valid_episode" in df.columns:
            df = df[pd.to_numeric(df["valid_episode"], errors="coerce").fillna(1).astype(int) == 1].copy()
            if df.empty:
                continue
        parts = p.parts
        algo = next((x for x in ["A2C", "PPO", "DQN", "RecurrentPPO"] if x in parts), df.get("algorithm", pd.Series(["unknown"])).iloc[0])
        seed_part = next((x for x in parts if x.startswith("seed_")), "seed_-1")
        seed = int(seed_part.split("_")[-1])
        eval_part = next((x for x in parts if x.startswith("eval_")), "eval_unknown")
        df["algorithm"] = algo
        df["seed_group"] = seed
        df["condition"] = eval_part.replace("eval_", "")
        df["source"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["forced_win", "unforced_win"]:
        if col not in df.columns:
            df[col] = 0
    for col in [
        "contact_ratio", "contact_forward_ratio", "front_visible_nonforward_ratio",
        "push_outward_progress_ratio", "empirical_action_entropy", "first_contact_events",
        "translational_forward_ratio", "straight_forward_ratio", "curved_forward_ratio",
        "reset_integrity_failure",
    ]:
        if col not in df.columns:
            df[col] = 0.0
    return (df.groupby(["algorithm", "seed_group", "condition"], as_index=False)
              .agg(win_rate=("win", "mean"), loss_rate=("loss", "mean"), draw_rate=("draw", "mean"),
                   forced_win_rate=("forced_win", "mean"), unforced_win_rate=("unforced_win", "mean"),
                   mean_reward=("episode_reward", "mean"), median_steps=("episode_steps", "median"),
                   aggression_index=("aggression_index", "mean"), engagement_ratio=("engagement_ratio", "mean"),
                   attack_ratio=("attack_ratio", "mean"), forward_ratio=("forward_ratio", "mean"),
                   translational_forward_ratio=("translational_forward_ratio", "mean"),
                   straight_forward_ratio=("straight_forward_ratio", "mean"),
                   curved_forward_ratio=("curved_forward_ratio", "mean"),
                   target_detected_ratio=("target_detected_ratio", "mean"),
                   search_forward_ratio=("search_forward_ratio", "mean"), scan_ratio=("scan_ratio", "mean"),
                   approach_progress_ratio=("approach_progress_ratio", "mean"),
                   edge_risk_ratio=("edge_risk_ratio", "mean"),
                   contact_ratio=("contact_ratio", "mean"),
                   contact_forward_ratio=("contact_forward_ratio", "mean"),
                   first_contact_events=("first_contact_events", "mean"),
                   front_visible_nonforward_ratio=("front_visible_nonforward_ratio", "mean"),
                   push_outward_progress_ratio=("push_outward_progress_ratio", "mean"),
                   empirical_action_entropy=("empirical_action_entropy", "mean")))


def overall_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["win_rate", "forced_win_rate", "unforced_win_rate", "mean_reward", "median_steps",
               "aggression_index", "engagement_ratio", "attack_ratio", "forward_ratio",
               "translational_forward_ratio", "straight_forward_ratio", "curved_forward_ratio",
               "target_detected_ratio", "search_forward_ratio", "scan_ratio",
               "approach_progress_ratio", "edge_risk_ratio", "contact_ratio",
               "contact_forward_ratio", "first_contact_events", "front_visible_nonforward_ratio",
               "push_outward_progress_ratio", "empirical_action_entropy"]
    for (algo, cond), g in seed_df.groupby(["algorithm", "condition"]):
        row = {"algorithm": algo, "condition": cond, "n_seeds": len(g)}
        for m in metrics:
            vals = g[m].to_numpy(float)
            row[f"{m}_mean"] = np.nanmean(vals)
            row[f"{m}_sd"] = np.nanstd(vals, ddof=1) if len(vals) > 1 else np.nan
            row[f"{m}_ci95"] = ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_tests(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, cdf in seed_df.groupby("condition"):
        algos = sorted(cdf["algorithm"].unique())
        groups = [cdf.loc[cdf.algorithm == a, "win_rate"].dropna().to_numpy() for a in algos]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            try:
                h, pkw = stats.kruskal(*groups)
            except ValueError:
                h, pkw = np.nan, np.nan
        else:
            h, pkw = np.nan, np.nan
        pairs, pvals, stats_u = [], [], []
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                x = groups[i]; y = groups[j]
                if len(x) and len(y):
                    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                else:
                    u, p = np.nan, np.nan
                pairs.append((algos[i], algos[j])); stats_u.append(u); pvals.append(p)
        adj = holm_adjust([p if np.isfinite(p) else 1.0 for p in pvals]) if pvals else []
        for k, (a, b) in enumerate(pairs):
            rows.append({"condition": cond, "metric": "win_rate", "kruskal_H": h, "kruskal_p": pkw,
                         "algorithm_a": a, "algorithm_b": b, "mannwhitney_U": stats_u[k],
                         "p_raw": pvals[k], "p_holm": adj[k]})
    return pd.DataFrame(rows)


def read_training(root: Path):
    frames = []
    for p in root.rglob("train_episodes_agent0.csv"):
        if "stage2_scripted" not in p.parts:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        if "valid_episode" in df.columns:
            df = df[pd.to_numeric(df["valid_episode"], errors="coerce").fillna(1).astype(int) == 1].copy()
            if df.empty:
                continue
        algo = next((x for x in ["A2C", "PPO", "DQN", "RecurrentPPO"] if x in p.parts), "unknown")
        seed_part = next((x for x in p.parts if x.startswith("seed_")), "seed_-1")
        df["algorithm"] = algo
        df["seed_group"] = int(seed_part.split("_")[-1])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paper_plots(eval_df, seed_df, train_df, out):
    out.mkdir(parents=True, exist_ok=True)
    algos = [a for a in ["A2C", "PPO", "DQN", "RecurrentPPO"] if a in seed_df.algorithm.unique()]

    for cond in sorted(seed_df.condition.unique()):
        g = seed_df[seed_df.condition == cond]
        means = [g.loc[g.algorithm == a, "win_rate"].mean() for a in algos]
        errs = [ci95(g.loc[g.algorithm == a, "win_rate"]) for a in algos]
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(algos, means, yerr=errs, capsize=4)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Win rate")
        ax.set_title(f"Algorithm comparison: {cond} opponent")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"win_rate_{cond}.png", dpi=300)
        fig.savefig(out / f"win_rate_{cond}.pdf")
        plt.close(fig)

    if not seed_df.empty:
        for cond in sorted(seed_df.condition.unique()):
            g = seed_df[seed_df.condition == cond]
            forced = [g.loc[g.algorithm == a, "forced_win_rate"].mean() for a in algos]
            unforced = [g.loc[g.algorithm == a, "unforced_win_rate"].mean() for a in algos]
            fig, ax = plt.subplots(figsize=(7.6, 4.8))
            x = np.arange(len(algos))
            ax.bar(x, forced, label="Contact-assisted wins")
            ax.bar(x, unforced, bottom=forced, label="Unforced opponent self-outs")
            ax.set_xticks(x, algos)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Win-rate contribution")
            ax.set_title(f"Win causality: {cond} opponent")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / f"win_causality_{cond}.png", dpi=300)
            fig.savefig(out / f"win_causality_{cond}.pdf")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        data = [seed_df.loc[seed_df.algorithm == a, "aggression_index"].dropna().to_numpy() for a in algos]
        ax.boxplot(data, tick_labels=algos)
        ax.set_ylabel("Aggression index")
        ax.set_title("Behavioral aggressiveness across evaluation conditions")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "aggression_algorithm_comparison.png", dpi=300)
        fig.savefig(out / "aggression_algorithm_comparison.pdf")
        plt.close(fig)

    if not seed_df.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        x = np.arange(len(algos))
        width = 0.20
        translation = [seed_df.loc[seed_df.algorithm == a, "translational_forward_ratio"].mean() for a in algos]
        attack = [seed_df.loc[seed_df.algorithm == a, "attack_ratio"].mean() for a in algos]
        contact = [seed_df.loc[seed_df.algorithm == a, "contact_ratio"].mean() for a in algos]
        pressure = [seed_df.loc[seed_df.algorithm == a, "contact_forward_ratio"].mean() for a in algos]
        ax.bar(x - 1.5 * width, translation, width, label="Forward translation")
        ax.bar(x - 0.5 * width, attack, width, label="Attack")
        ax.bar(x + 0.5 * width, contact, width, label="Physical contact")
        ax.bar(x + 1.5 * width, pressure, width, label="Translational pressure")
        ax.set_xticks(x, algos)
        ax.set_ylabel("Mean episode ratio")
        ax.set_title("Attack-learning and translational-motion audit")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "attack_contact_behavior.png", dpi=300)
        fig.savefig(out / "attack_contact_behavior.pdf")
        plt.close(fig)

    if not train_df.empty:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for algo in algos:
            adf = train_df[train_df.algorithm == algo].copy()
            if adf.empty:
                continue
            max_step = int(adf.total_env_steps.max())
            bins = np.linspace(0, max_step, 60)
            adf["bin"] = pd.cut(adf.total_env_steps, bins=bins, include_lowest=True, labels=False)
            curve = adf.groupby("bin", as_index=False).agg(step=("total_env_steps", "mean"), win=("win", "mean"))
            ax.plot(curve.step, curve.win, label=algo)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Binned training win rate")
        ax.set_ylim(0, 1)
        ax.set_title("Learning curves during moving-opponent training")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "learning_curve_win_rate.png", dpi=300)
        fig.savefig(out / "learning_curve_win_rate.pdf")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    out = root / "paper_results"
    out.mkdir(parents=True, exist_ok=True)

    eval_df = read_evaluations(root)
    if eval_df.empty:
        print("No evaluation.csv files found")
        return
    seed_df = seed_summary(eval_df)
    overall = overall_summary(seed_df)
    tests = pairwise_tests(seed_df)
    train_df = read_training(root)

    eval_df.to_csv(out / "all_evaluation_episodes.csv", index=False)
    seed_df.to_csv(out / "per_seed_summary.csv", index=False)
    overall.to_csv(out / "algorithm_summary_mean_sd_ci95.csv", index=False)
    tests.to_csv(out / "statistical_tests_win_rate.csv", index=False)
    paper_plots(eval_df, seed_df, train_df, out / "figures")
    print(f"Analysis written to {out}")


if __name__ == "__main__":
    main()
