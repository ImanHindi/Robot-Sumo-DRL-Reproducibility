#!/usr/bin/env python3
"""
Updated reproducible figure generator for the manuscript:

    Competitive Learning and Adaptation in Autonomous Robot-Sumo:
    Multi-Seed Deep Reinforcement Learning Benchmarking and
    Asymmetric Veteran-Novice Self-Play

This version fixes clarity and consistency issues noted during figure review:
- uses explicit and non-overlapping color palettes;
- explains seed-level dots via in-figure annotation;
- avoids misleading line connections across unordered seeds;
- adds 95% confidence intervals consistently for seed-level summaries;
- improves the Stage-2 learning curve by averaging per seed first;
- improves self-play figures so their axes and data structure are explicit.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Wedge, Polygon
import numpy as np
import pandas as pd
from scipy import stats

ALGORITHMS = ["A2C", "PPO", "DQN", "RecurrentPPO"]
CONDITIONS = ["static", "scripted", "aggressive"]
CONDITION_LABELS = {"static": "Static", "scripted": "Scripted", "aggressive": "Aggressive"}
CONDITION_COLORS = {"static": "#4C78A8", "scripted": "#F58518", "aggressive": "#E45756"}
ALGO_COLORS = {"A2C": "#4C78A8", "PPO": "#F58518", "DQN": "#54A24B", "RecurrentPPO": "#B279A2"}
OUTCOME_COLORS = {"win": "#2CA02C", "draw": "#BDBDBD", "loss": "#D62728"}
METRIC_COLORS = {
    "translational_forward_ratio": "#4C78A8",
    "attack_ratio": "#F58518",
    "contact_ratio": "#54A24B",
    "contact_forward_ratio": "#E45756",
}


def save_figure(fig: plt.Figure, out_dir: Path, number: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"Figure_{number:02d}"
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci95_t(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, 0.0
    return float(np.mean(x)), ci95_t(x)


def require_columns(df: pd.DataFrame, names: Iterable[str], source: Path) -> None:
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def read_csv_required(path: Path, required: Iterable[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_columns(df, required, path)
    return df


def add_summary_note(ax, text: str) -> None:
    ax.text(
        0.01, 0.99, text,
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9),
    )


# ---------------------------------------------------------------------------
# Fig. 1 - Experimental workflow
# ---------------------------------------------------------------------------

def figure_01_workflow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.9, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Experimental workflow and evaluation design", fontsize=12, pad=12)

    def box(x, y, w, h, text, fs=8):
        r = Rectangle((x, y), w, h, fill=False, linewidth=1.1)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
        return (x, y, w, h)

    def arrow(a, b):
        x1, y1, w1, h1 = a
        x2, y2, w2, h2 = b
        p1 = (x1 + w1, y1 + h1 / 2)
        p2 = (x2, y2 + h2 / 2)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=12,
                                     linewidth=1.0, connectionstyle="arc3"))

    b1 = box(0.45, 5.25, 2.0, 1.0, "Official 3 kg\nRobot-Sumo model\n(Webots)")
    b2 = box(3.1, 5.25, 2.0, 1.0, "Hardware-reproducible\n6-D observation\n18 discrete actions")
    b3 = box(5.75, 5.25, 2.0, 1.0, "Stage 1 curriculum\n200k steps\nstatic opponent")
    b4 = box(8.4, 5.25, 2.0, 1.0, "Stage 2\n300k steps\nscripted opponent")
    arrow(b1, b2); arrow(b2, b3); arrow(b3, b4)

    b5 = box(1.55, 2.95, 2.2, 1.0, "4 RL algorithms\nA2C, PPO, DQN,\nRecurrentPPO")
    b6 = box(4.45, 2.95, 2.2, 1.0, "5 independent seeds\n500k steps each\n20 trained policies")
    b7 = box(7.35, 2.95, 2.2, 1.0, "Frozen evaluation\n100 × static\n100 × scripted\n100 × aggressive")
    arrow(b5, b6); arrow(b6, b7)

    ax.add_patch(FancyArrowPatch((6.7, 5.25), (2.65, 3.95), arrowstyle="->",
                                 mutation_scale=12, linewidth=1.0))
    ax.add_patch(FancyArrowPatch((9.4, 5.25), (8.45, 3.95), arrowstyle="->",
                                 mutation_scale=12, linewidth=1.0))

    box(4.45, 1.0, 3.4, 1.05,
        "Seed-level statistical analysis\nwin/loss/forced-win causality and\nbehavioral + safety metrics")
    ax.add_patch(FancyArrowPatch((8.45, 2.95), (6.15, 2.05), arrowstyle="->",
                                 mutation_scale=12, linewidth=1.0))

    b9 = box(8.65, 0.75, 2.6, 1.55,
             "Asymmetric self-play extension\nPPO veterans + fresh novices\n500k Stage-3 steps\n100k-step checkpoints", fs=7.5)
    ax.add_patch(FancyArrowPatch((6.15, 1.0), (8.65, 1.52), arrowstyle="->",
                                 mutation_scale=12, linewidth=1.0))

    save_figure(fig, out_dir, 1)


# ---------------------------------------------------------------------------
# Fig. 2 - Arena, robot, and sensors
# ---------------------------------------------------------------------------

def figure_02_arena_robot_sensor_schematic(out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.45))
    fig.suptitle("Frozen v1.12 Webots Robot-Sumo physical and sensing model", fontsize=10)

    # colored schematic for better readability in the manuscript
    dohyo_border = "#2F4B7C"
    dohyo_inner = "#E8EEF7"
    shikiri_color = "#6C8EBF"
    spawn_box_color = "#8FAADC"
    chassis_fill = "#DCE6F2"
    chassis_edge = "#2F4B7C"
    wedge_fill = "#F6D9A6"
    wedge_edge = "#C98E2B"
    wheel_fill = "#8FB9E1"
    wheel_edge = "#2E5F87"
    tof_fill = "#B9D7EA"
    tof_edge = "#287D8E"
    tof_beam = "#6CC3D5"
    edge_sensor_fill = "#F4A7A1"
    edge_sensor_edge = "#B85450"
    annotation_color = "#404040"
    ground_color = "#555555"
    com_color = "#8B0000"

    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Official-geometry Dohyo", fontsize=8)
    outer = Circle((0, 0), 0.77, facecolor="white", edgecolor=dohyo_border, linewidth=1.3)
    inner = Circle((0, 0), 0.72, facecolor=dohyo_inner, edgecolor=dohyo_border, linewidth=0.9)
    ax.add_patch(outer)
    ax.add_patch(inner)
    for x in (-0.10, +0.10):
        ax.add_patch(Rectangle((x - 0.01, -0.10), 0.02, 0.20, facecolor=shikiri_color, edgecolor=dohyo_border, linewidth=0.6))
    for x, y, angle in ((-0.22, 0.12, 25), (0.24, -0.14, -35)):
        sq = Rectangle((x - 0.0975, y - 0.0975), 0.195, 0.195, fill=False, linewidth=0.8,
                       edgecolor=spawn_box_color, linestyle="--", angle=angle)
        ax.add_patch(sq)
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.9, 0.9)
    ax.text(0, -0.86, "1.54 m outer diameter; 5 cm white boundary", ha="center", fontsize=6, color=annotation_color)

    ax = axes[1]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) Top view and sensor layout", fontsize=8)
    ax.add_patch(Rectangle((-0.0975, -0.0975), 0.195, 0.195, fill=False, linewidth=0.9,
                           linestyle="--", edgecolor=spawn_box_color))
    ax.add_patch(Rectangle((-0.095, -0.0625), 0.190, 0.125, facecolor=chassis_fill, edgecolor=chassis_edge, linewidth=0.9))
    wedge = Polygon([(-0.0975, -0.0625), (0.0975, -0.0625), (0.0975, -0.0975), (-0.0975, -0.0975)],
                    closed=True, facecolor=wedge_fill, edgecolor=wedge_edge, linewidth=0.8)
    ax.add_patch(wedge)
    for x in (-0.073, +0.073):
        ax.add_patch(Rectangle((x - 0.014, 0.028 - 0.030), 0.028, 0.060, facecolor=wheel_fill, edgecolor=wheel_edge, linewidth=0.7))
    fans = [
        ((0.0, -0.091), -90, 0.42, 0.85),
        ((0.0, +0.094), +90, 0.50, 0.65),
        ((+0.094, +0.010), 0, 0.52, 0.70),
        ((-0.094, +0.010), 180, 0.52, 0.70),
    ]
    scale = 0.20
    for (x, y), center_deg, aperture, _rng in fans:
        half_deg = np.degrees(aperture) / 2
        ax.add_patch(Wedge((x, y), scale, center_deg - half_deg, center_deg + half_deg,
                           alpha=0.18, edgecolor=tof_beam, facecolor=tof_beam, linewidth=0.7))
        ax.plot([x], [y], marker="o", markersize=3.0, markerfacecolor=tof_fill,
                markeredgecolor=tof_edge, color=tof_edge)
    for x in (-0.078, +0.078):
        for y in (-0.075, +0.075):
            ax.plot(x, y, marker="s", markersize=3.6, markerfacecolor=edge_sensor_fill,
                    markeredgecolor=edge_sensor_edge, color=edge_sensor_edge)
    # Separate annotations to avoid overlap in the final manuscript rendering.
    label_box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.90)
    ax.annotate("front / local −Y", xy=(0.0, -0.205), xytext=(0.105, -0.265),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=annotation_color),
                ha="center", va="center", fontsize=5.8, color=annotation_color, bbox=label_box)
    ax.text(-0.31, 0.255, "195 × 195 mm envelope\n190 × 125 × 48 mm chassis\n60 mm wheels; 146 mm track",
            fontsize=5.4, va="top", ha="left", color=annotation_color, bbox=label_box)
    ax.annotate("4 downward edge sensors\n(x=±78 mm, y=±75 mm)",
                xy=(-0.078, -0.075), xytext=(-0.315, -0.185),
                arrowprops=dict(arrowstyle="->", lw=0.65, color=edge_sensor_edge),
                fontsize=5.3, ha="left", va="top", color=edge_sensor_edge, bbox=label_box)
    ax.text(0.17, 0.255, "ToF sensors + FOV", fontsize=5.4, va="top", ha="left",
            color=tof_edge, bbox=label_box)
    ax.set_xlim(-0.33, 0.33)
    ax.set_ylim(-0.31, 0.29)

    ax = axes[2]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) Side-view mechanical reference", fontsize=8)
    ground_y = 0.0
    ax.plot([-0.18, 0.18], [ground_y, ground_y], linewidth=0.9, color=ground_color)
    ax.add_patch(Rectangle((-0.0625, 0.030), 0.125, 0.048, facecolor=chassis_fill, edgecolor=chassis_edge, linewidth=0.9))
    for x in (-0.035, +0.035):
        ax.add_patch(Circle((x, 0.030), 0.030, facecolor=wheel_fill, edgecolor=wheel_edge, linewidth=0.7))
    ax.add_patch(Polygon([(-0.0625, 0.030), (-0.135, 0.0025), (-0.135, 0.005), (-0.0625, 0.040)],
                         closed=True, facecolor=wedge_fill, edgecolor=wedge_edge, linewidth=0.8))
    ax.plot(0.015, 0.054, marker="x", markersize=5.5, color=com_color, markeredgewidth=1.0)
    ax.text(0.020, 0.059, "ref. CoM", fontsize=5.6, color=com_color)
    ax.annotate("60 mm wheel", xy=(0.035, 0.030), xytext=(0.075, 0.090),
                arrowprops=dict(arrowstyle="->", lw=0.6, color=annotation_color), fontsize=5.5, color=annotation_color)
    ax.annotate("2.5 mm wedge\n13.8°", xy=(-0.115, 0.010), xytext=(-0.17, 0.105),
                arrowprops=dict(arrowstyle="->", lw=0.6, color=annotation_color), fontsize=5.5, color=annotation_color)
    ax.annotate("edge sensor\n≈2.6 mm clearance", xy=(-0.048, 0.004), xytext=(-0.01, -0.050),
                arrowprops=dict(arrowstyle="->", lw=0.6, color=annotation_color), fontsize=5.5, color=annotation_color)
    ax.set_xlim(-0.19, 0.19)
    ax.set_ylim(-0.065, 0.135)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_dir, 2)


# ---------------------------------------------------------------------------
# Benchmark data utilities for Figs. 3-10
# ---------------------------------------------------------------------------

def read_stage2_training(benchmark_root: Path) -> pd.DataFrame:
    frames = []
    for p in benchmark_root.rglob("train_episodes_agent0.csv"):
        if "stage2_scripted" not in p.parts:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "total_env_steps" not in df.columns or "win" not in df.columns:
            continue
        if "valid_episode" in df.columns:
            valid = pd.to_numeric(df["valid_episode"], errors="coerce").fillna(1).astype(int) == 1
            df = df.loc[valid].copy()
        algo = next((a for a in ALGORITHMS if a in p.parts), None)
        seed_part = next((x for x in p.parts if x.startswith("seed_")), None)
        if algo is None or seed_part is None:
            continue
        df["algorithm"] = algo
        df["seed_group"] = int(seed_part.split("_")[-1])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_benchmark_paper_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_path = path / "per_seed_summary.csv"
    eval_path = path / "all_evaluation_episodes.csv"
    seed_df = read_csv_required(
        seed_path,
        ["algorithm", "seed_group", "condition", "win_rate", "loss_rate", "draw_rate",
         "forced_win_rate", "unforced_win_rate", "aggression_index"],
    )
    eval_df = read_csv_required(eval_path, ["algorithm", "seed_group", "condition", "win", "loss", "draw"])

    # Backward-compatible aliases if needed
    aliases = {
        "forward_ratio": "translational_forward_ratio",
    }
    for old, new in aliases.items():
        if new not in seed_df.columns and old in seed_df.columns:
            seed_df[new] = seed_df[old]

    for col in ["attack_ratio", "translational_forward_ratio", "contact_ratio", "contact_forward_ratio"]:
        if col not in seed_df.columns:
            seed_df[col] = np.nan

    return seed_df, eval_df


def figure_03_learning_curve(out_dir: Path, benchmark_root: Path) -> None:
    train_df = read_stage2_training(benchmark_root)
    if train_df.empty:
        raise FileNotFoundError("Fig. 3 requires raw Stage-2 train_episodes_agent0.csv files under --benchmark-root.")

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for algo in ALGORITHMS:
        adf = train_df[train_df["algorithm"] == algo].copy()
        if adf.empty:
            continue
        adf["total_env_steps"] = pd.to_numeric(adf["total_env_steps"], errors="coerce")
        adf["win"] = pd.to_numeric(adf["win"], errors="coerce")
        max_step = int(np.nanmax(adf["total_env_steps"]))
        bins = np.linspace(0, max_step, 31)
        adf["bin"] = pd.cut(adf["total_env_steps"], bins=bins, include_lowest=True, labels=False)
        per_seed = adf.groupby(["seed_group", "bin"], as_index=False).agg(
            step=("total_env_steps", "mean"),
            win=("win", "mean"),
        )
        summary = per_seed.groupby("bin", as_index=False).agg(step=("step", "mean"), win_mean=("win", "mean"))
        summary["win_ci"] = per_seed.groupby("bin")["win"].apply(ci95_t).values
        ax.plot(summary["step"], summary["win_mean"], label=algo, color=ALGO_COLORS.get(algo))
        ax.fill_between(summary["step"], summary["win_mean"] - summary["win_ci"], summary["win_mean"] + summary["win_ci"],
                        color=ALGO_COLORS.get(algo), alpha=0.15)
    ax.set_xlabel("Stage-2 environment steps")
    ax.set_ylabel("Mean training win rate")
    ax.set_ylim(0, 0.45)
    ax.set_title("Learning curves during scripted-opponent training")
    ax.grid(alpha=0.25)
    ax.legend(title="Algorithm")
    add_summary_note(ax, "Lines = mean across five seeds\nShaded bands = 95% CI")
    fig.tight_layout()
    save_figure(fig, out_dir, 3)


def figure_04_seed_winrates(out_dir: Path, seed_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    algos = [a for a in ALGORITHMS if a in set(seed_df["algorithm"])]
    x = np.arange(len(algos), dtype=float)
    width = 0.22
    point_offsets = np.array([-0.035, -0.017, 0.0, 0.017, 0.035])

    for j, cond in enumerate(CONDITIONS):
        xpos = x + (j - 1) * width
        means, errs = [], []
        for algo in algos:
            vals = seed_df[(seed_df.algorithm == algo) & (seed_df.condition == cond)]["win_rate"].to_numpy(float)
            m, e = mean_ci(vals)
            means.append(m); errs.append(e)
        ax.bar(xpos, means, width=width, yerr=errs, capsize=4, label=CONDITION_LABELS[cond],
               color=CONDITION_COLORS[cond], alpha=0.85, edgecolor="black", linewidth=0.5)
        for i, algo in enumerate(algos):
            sub = seed_df[(seed_df.algorithm == algo) & (seed_df.condition == cond)].sort_values("seed_group")
            ys = sub["win_rate"].to_numpy(float)
            offs = point_offsets[:len(ys)] if len(ys) <= len(point_offsets) else np.linspace(-0.04, 0.04, len(ys))
            ax.scatter(np.full(len(ys), xpos[i]) + offs, ys, s=34, marker="o",
                       facecolors="white", edgecolors=CONDITION_COLORS[cond], linewidths=1.2, zorder=4)
    ax.set_xticks(x, algos)
    ax.set_ylabel("Evaluation win rate")
    ax.set_ylim(0, 0.45)
    ax.set_title("Evaluation win rate across five training seeds")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Evaluation condition", ncol=3, frameon=True)
    add_summary_note(ax, "Bars = seed mean\nError bars = 95% CI\nOpen circles = individual seeds")
    fig.tight_layout()
    save_figure(fig, out_dir, 4)


def figure_05_aggressive_winrate(out_dir: Path, seed_df: pd.DataFrame) -> None:
    g = seed_df[seed_df.condition == "aggressive"].copy()
    algos = [a for a in ALGORITHMS if a in set(g.algorithm)]
    means, errs = [], []
    for algo in algos:
        m, e = mean_ci(g.loc[g.algorithm == algo, "win_rate"])
        means.append(m); errs.append(e)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(algos))
    colors = [ALGO_COLORS[a] for a in algos]
    ax.bar(x, means, yerr=errs, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    for i, algo in enumerate(algos):
        ys = g.loc[g.algorithm == algo, "win_rate"].to_numpy(float)
        offs = np.linspace(-0.08, 0.08, len(ys))
        ax.scatter(np.full(len(ys), x[i]) + offs, ys, facecolors="white", edgecolors=ALGO_COLORS[algo], s=35, zorder=4)
    ax.set_xticks(x, algos)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("Win rate")
    ax.set_title("Aggressive-opponent evaluation win rate")
    ax.grid(axis="y", alpha=0.25)
    add_summary_note(ax, "Bars = mean across seeds\nError bars = 95% CI\nOpen circles = individual seeds")
    fig.tight_layout()
    save_figure(fig, out_dir, 5)


def figure_06_win_attribution(out_dir: Path, seed_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.8), sharey=True)
    algos = [a for a in ALGORITHMS if a in set(seed_df.algorithm)]
    for ax, cond in zip(axes, CONDITIONS):
        g = seed_df[seed_df.condition == cond]
        forced = [g.loc[g.algorithm == a, "forced_win_rate"].mean() for a in algos]
        unforced = [g.loc[g.algorithm == a, "unforced_win_rate"].mean() for a in algos]
        x = np.arange(len(algos))
        ax.bar(x, forced, color="#4C78A8", label="Forced / contact-assisted wins", edgecolor="black", linewidth=0.4)
        ax.bar(x, unforced, bottom=forced, color="#A0CBE8", label="Unforced wins", edgecolor="black", linewidth=0.4)
        for i, total in enumerate(np.asarray(forced) + np.asarray(unforced)):
            if total > 0:
                ax.text(i, total + 0.006, f"{total:.02f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, algos, rotation=25, ha="right")
        ax.set_title(CONDITION_LABELS[cond], fontsize=10)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Contribution to win rate")
    axes[0].set_ylim(0, 0.16)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Attribution of evaluation wins", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_dir, 6)


def figure_07_outcome_composition(out_dir: Path, seed_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.9), sharey=True)
    algos = [a for a in ALGORITHMS if a in set(seed_df.algorithm)]
    for ax, cond in zip(axes, CONDITIONS):
        g = seed_df[seed_df.condition == cond]
        win = [g.loc[g.algorithm == a, "win_rate"].mean() for a in algos]
        draw = [g.loc[g.algorithm == a, "draw_rate"].mean() for a in algos]
        loss = [g.loc[g.algorithm == a, "loss_rate"].mean() for a in algos]
        x = np.arange(len(algos))
        ax.bar(x, win, label="Win", color=OUTCOME_COLORS["win"], edgecolor="black", linewidth=0.4)
        ax.bar(x, draw, bottom=win, label="Draw", color=OUTCOME_COLORS["draw"], edgecolor="black", linewidth=0.4)
        ax.bar(x, loss, bottom=np.asarray(win) + np.asarray(draw), label="Loss", color=OUTCOME_COLORS["loss"], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, algos, rotation=25, ha="right")
        ax.set_title(CONDITION_LABELS[cond], fontsize=10)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean proportion of evaluation outcomes")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Outcome composition across 500 evaluations per algorithm-condition", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_dir, 7)


def figure_08_aggression(out_dir: Path, seed_df: pd.DataFrame) -> None:
    algos = [a for a in ALGORITHMS if a in set(seed_df.algorithm)]
    data = [seed_df.loc[seed_df.algorithm == a, "aggression_index"].dropna().to_numpy(float) for a in algos]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bp = ax.boxplot(data, tick_labels=algos, patch_artist=True)
    for patch, algo in zip(bp["boxes"], algos):
        patch.set_facecolor(ALGO_COLORS[algo])
        patch.set_alpha(0.25)
    for i, algo in enumerate(algos, start=1):
        ys = seed_df.loc[seed_df.algorithm == algo, "aggression_index"].dropna().to_numpy(float)
        offs = np.linspace(-0.10, 0.10, len(ys))
        ax.scatter(np.full(len(ys), i) + offs, ys, s=16, color=ALGO_COLORS[algo], alpha=0.7, zorder=4)
    ax.set_ylabel("Aggression index")
    ax.set_title("Behavioral aggressiveness across all evaluation conditions")
    ax.grid(axis="y", alpha=0.25)
    add_summary_note(ax, "Each box pools all seed-condition summaries for one algorithm")
    fig.tight_layout()
    save_figure(fig, out_dir, 8)


def figure_09_attack_contact(out_dir: Path, seed_df: pd.DataFrame) -> None:
    algos = [a for a in ALGORITHMS if a in set(seed_df.algorithm)]
    x = np.arange(len(algos))
    width = 0.18
    metrics = [
        ("translational_forward_ratio", "Forward translation"),
        ("attack_ratio", "Attack"),
        ("contact_ratio", "Physical contact"),
        ("contact_forward_ratio", "Translational pressure"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for j, (col, label) in enumerate(metrics):
        means, errs = [], []
        for algo in algos:
            vals = seed_df.loc[seed_df.algorithm == algo, col].to_numpy(float)
            m, e = mean_ci(vals)
            means.append(m); errs.append(e)
        ax.bar(x + (j - 1.5) * width, means, width, yerr=errs, capsize=3, label=label,
               color=METRIC_COLORS[col], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, algos)
    ax.set_ylabel("Mean episode ratio")
    ax.set_title("Attack-learning and translational-motion audit")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    add_summary_note(ax, "Means and 95% CI across all seed-condition summaries")
    fig.tight_layout()
    save_figure(fig, out_dir, 9)


def figure_10_seed_heatmap(out_dir: Path, seed_df: pd.DataFrame) -> None:
    rows = []
    labels = []
    for algo in ALGORITHMS:
        for seed in sorted(seed_df.loc[seed_df.algorithm == algo, "seed_group"].unique()):
            vals = []
            for cond in CONDITIONS:
                s = seed_df[(seed_df.algorithm == algo) & (seed_df.seed_group == seed) & (seed_df.condition == cond)]
                vals.append(float(s.win_rate.iloc[0]) if len(s) else np.nan)
            rows.append(vals)
            labels.append(f"{algo} s{int(seed)}")
    mat = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(7.9, 8.4))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=max(0.4, np.nanmax(mat)))
    ax.set_xticks(np.arange(3), [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    ax.set_title("Seed-level win-rate variability")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                color = "white" if mat[i, j] > 0.22 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Win rate")
    fig.tight_layout()
    save_figure(fig, out_dir, 10)


# ---------------------------------------------------------------------------
# Self-play Figs. 11-14
# ---------------------------------------------------------------------------

def load_selfplay_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h2h = read_csv_required(
        path / "selfplay_head_to_head_per_seed_checkpoint.csv",
        ["seed", "stage3_steps", "veteran_win_rate", "novice_win_rate", "draw_rate",
         "veteran_forced_win_rate", "novice_forced_win_rate",
         "veteran_attack_ratio", "novice_attack_ratio",
         "veteran_contact_ratio", "novice_contact_ratio"],
    )
    summary = read_csv_required(
        path / "selfplay_head_to_head_summary_ci95.csv",
        ["stage3_steps", "veteran_win_rate_mean", "novice_win_rate_mean", "draw_rate_mean",
         "veteran_win_rate_ci95_low", "veteran_win_rate_ci95_high",
         "novice_win_rate_ci95_low", "novice_win_rate_ci95_high"],
    )
    external = read_csv_required(
        path / "selfplay_external_generalization_per_seed.csv",
        ["seed", "role", "opponent", "forced_win_rate"],
    )
    return h2h, summary, external


def figure_11_selfplay_official_trajectory(out_dir: Path, summary: pd.DataFrame) -> None:
    x = summary.stage3_steps.to_numpy(float) / 1000.0
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.errorbar(x, summary["veteran_win_rate_mean"],
                yerr=np.vstack([
                    summary["veteran_win_rate_mean"] - summary["veteran_win_rate_ci95_low"],
                    summary["veteran_win_rate_ci95_high"] - summary["veteran_win_rate_mean"],
                ]),
                marker="o", capsize=3, label="Veteran", color="#1F77B4")
    ax.errorbar(x, summary["novice_win_rate_mean"],
                yerr=np.vstack([
                    summary["novice_win_rate_mean"] - summary["novice_win_rate_ci95_low"],
                    summary["novice_win_rate_ci95_high"] - summary["novice_win_rate_mean"],
                ]),
                marker="s", capsize=3, label="Novice", color="#FF7F0E")
    ax.plot(x, summary["draw_rate_mean"], marker="^", label="Draw", color="#2CA02C")
    ax.set_xlabel("Stage-3 self-play steps (×10³)")
    ax.set_ylabel("Official head-to-head rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_title("Asymmetric self-play trajectory")
    ax.legend()
    ax.grid(alpha=0.25)
    add_summary_note(ax, "Veteran/novice error bars = 95% CI across seeds")
    fig.tight_layout()
    save_figure(fig, out_dir, 11)


def figure_12_selfplay_final_advantage(out_dir: Path, h2h: pd.DataFrame) -> None:
    final_step = h2h.stage3_steps.max()
    f = h2h[h2h.stage3_steps == final_step].sort_values("seed")
    official = f.novice_win_rate.to_numpy(float) - f.veteran_win_rate.to_numpy(float)
    forced = f.novice_forced_win_rate.to_numpy(float) - f.veteran_forced_win_rate.to_numpy(float)
    seeds = f.seed.to_numpy(int)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    x = np.arange(len(seeds), dtype=float)
    ax.axhline(0, linewidth=1, color="0.35")
    ax.scatter(x - 0.10, official, s=55, marker="o", color="#1F77B4", label="Official win-rate difference")
    ax.scatter(x + 0.10, forced, s=55, marker="s", color="#FF7F0E", label="Forced-win-rate difference")
    for xi, yi in zip(x - 0.10, official):
        ax.plot([xi, xi], [0, yi], color="#1F77B4", alpha=0.4, linewidth=1)
    for xi, yi in zip(x + 0.10, forced):
        ax.plot([xi, xi], [0, yi], color="#FF7F0E", alpha=0.4, linewidth=1)
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Novice − veteran rate at final checkpoint")
    ax.set_title("Final self-play seed-wise advantage (500k Stage-3 steps)")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save_figure(fig, out_dir, 12)


def figure_13_selfplay_behavior(out_dir: Path, h2h: pd.DataFrame) -> None:
    steps = sorted(h2h.stage3_steps.unique())
    x = np.asarray(steps, dtype=float) / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)

    panels = [
        (axes[0], "attack", "Attack ratio"),
        (axes[1], "contact", "Contact ratio"),
    ]
    for ax, stem, title in panels:
        vet_means, vet_cis = [], []
        nov_means, nov_cis = [], []
        for s in steps:
            sub = h2h[h2h.stage3_steps == s]
            vm, vc = mean_ci(sub[f"veteran_{stem}_ratio"])
            nm, nc = mean_ci(sub[f"novice_{stem}_ratio"])
            vet_means.append(vm); vet_cis.append(vc)
            nov_means.append(nm); nov_cis.append(nc)
        ax.errorbar(x, vet_means, yerr=vet_cis, marker="o", capsize=3, label="Veteran", color="#1F77B4")
        ax.errorbar(x, nov_means, yerr=nov_cis, marker="s", capsize=3, label="Novice", color="#FF7F0E")
        ax.set_title(title)
        ax.set_xlabel("Stage-3 self-play steps (×10³)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Behavioral ratio")
    axes[0].legend()
    fig.suptitle("Behavioral trajectory during asymmetric self-play", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_dir, 13)


def figure_14_external_generalization(out_dir: Path, external: pd.DataFrame) -> None:
    """External generalization using the exact transformation/style of the verified manuscript."""
    agg = external.groupby(["opponent", "role"], as_index=False).forced_win_rate.mean()
    roles = ["veteran_pre", "veteran_post", "novice_post"]
    labels = ["Veteran Pre", "Veteran Post", "Novice Post"]
    fig, ax = plt.subplots(figsize=(7.07, 4.50))
    x = np.arange(len(CONDITIONS))
    for role, label in zip(roles, labels):
        vals = []
        for cond in CONDITIONS:
            s = agg[(agg.opponent == cond) & (agg.role == role)]
            vals.append(float(s.forced_win_rate.iloc[0]) if len(s) else np.nan)
        ax.plot(x, vals, marker="o", label=label)
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
    ax.set_xlabel("External opponent")
    ax.set_ylabel("Forced win rate")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_dir, 14)

def validate_manuscript_datasets(seed_df: pd.DataFrame, eval_df: pd.DataFrame,
                                 h2h: pd.DataFrame, summary: pd.DataFrame,
                                 external: pd.DataFrame) -> None:
    """Fail fast if the wrong/partial result folder is supplied."""
    expected_algos = set(ALGORITHMS)
    expected_conditions = set(CONDITIONS)
    expected_seeds = {0, 1, 2, 3, 4}
    expected_steps = {100000, 200000, 300000, 400000, 500000}
    expected_roles = {"veteran_pre", "veteran_post", "novice_post"}

    if set(seed_df.algorithm.unique()) != expected_algos:
        raise ValueError(f"Benchmark algorithms mismatch: {sorted(seed_df.algorithm.unique())}")
    if set(seed_df.condition.unique()) != expected_conditions:
        raise ValueError(f"Benchmark conditions mismatch: {sorted(seed_df.condition.unique())}")
    if set(seed_df.seed_group.astype(int).unique()) != expected_seeds:
        raise ValueError(f"Benchmark seeds mismatch: {sorted(seed_df.seed_group.unique())}")
    if len(seed_df) != 4 * 5 * 3:
        raise ValueError(f"Expected 60 benchmark seed-condition rows, found {len(seed_df)}")
    if len(eval_df) != 6000:
        raise ValueError(f"Expected 6000 benchmark evaluation episodes, found {len(eval_df)}")

    if set(h2h.seed.astype(int).unique()) != expected_seeds:
        raise ValueError(f"Self-play seeds mismatch: {sorted(h2h.seed.unique())}")
    if set(h2h.stage3_steps.astype(int).unique()) != expected_steps:
        raise ValueError(f"Self-play checkpoints mismatch: {sorted(h2h.stage3_steps.unique())}")
    if len(h2h) != 25:
        raise ValueError(f"Expected 25 self-play seed-checkpoint rows, found {len(h2h)}")
    if set(summary.stage3_steps.astype(int).unique()) != expected_steps or len(summary) != 5:
        raise ValueError("Self-play summary must contain exactly the five 100k–500k checkpoints")

    if set(external.seed.astype(int).unique()) != expected_seeds:
        raise ValueError(f"External-generalization seeds mismatch: {sorted(external.seed.unique())}")
    if set(external.role.unique()) != expected_roles:
        raise ValueError(f"External-generalization roles mismatch: {sorted(external.role.unique())}")
    if set(external.opponent.unique()) != expected_conditions:
        raise ValueError(f"External-generalization conditions mismatch: {sorted(external.opponent.unique())}")
    if len(external) != 45:
        raise ValueError(f"Expected 45 external-generalization rows, found {len(external)}")

    # Exact five-seed means used by Fig. 14 in the verified manuscript.
    expected_fig14 = {
        ("veteran_pre", "static"): 0.022,
        ("veteran_pre", "scripted"): 0.102,
        ("veteran_pre", "aggressive"): 0.118,
        ("veteran_post", "static"): 0.002,
        ("veteran_post", "scripted"): 0.014,
        ("veteran_post", "aggressive"): 0.022,
        ("novice_post", "static"): 0.004,
        ("novice_post", "scripted"): 0.046,
        ("novice_post", "aggressive"): 0.040,
    }
    for (role, cond), expected in expected_fig14.items():
        actual = float(external[(external.role == role) & (external.opponent == cond)].forced_win_rate.mean())
        if not np.isclose(actual, expected, atol=1e-12):
            raise ValueError(
                f"Fig. 14 data mismatch for {role}/{cond}: expected mean {expected:.3f}, got {actual:.3f}. "
                "This usually means the wrong self-play result folder was supplied."
            )

    print("[dataset audit] PASS")
    print("[dataset audit] benchmark: 4 algorithms × 5 seeds × 3 conditions = 60 summaries; 6000 episodes")
    print("[dataset audit] self-play: 5 seeds × 5 checkpoints = 25 rows")
    print("[dataset audit] external generalization: 5 seeds × 3 roles × 3 opponents = 45 rows")
    print("[dataset audit] Fig. 14 five-seed means:")
    for role in ["veteran_pre", "veteran_post", "novice_post"]:
        vals=[]
        for cond in CONDITIONS:
            m=float(external[(external.role==role)&(external.opponent==cond)].forced_win_rate.mean())
            vals.append(f"{cond}={m:.3f}")
        print(f"  {role}: " + ", ".join(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 14 manuscript figures (updated version).")
    parser.add_argument("--benchmark-root", type=Path, help="Full frozen-v1.12 benchmark root (required for Fig. 3).")
    parser.add_argument("--benchmark-paper-results", type=Path, help="v1.12 paper_results directory; defaults to <benchmark-root>/paper_results.")
    parser.add_argument("--selfplay-results", type=Path, required=True, help="v1.14 self-play results directory containing the selfplay_*.csv files.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for Figure_01 ... Figure_14.")
    parser.add_argument("--skip-figure3", action="store_true", help="Skip Fig. 3 if raw benchmark training logs are unavailable.")
    args = parser.parse_args()

    benchmark_results = args.benchmark_paper_results
    if benchmark_results is None:
        if args.benchmark_root is None:
            parser.error("Provide --benchmark-root or --benchmark-paper-results.")
        benchmark_results = args.benchmark_root / "paper_results"

    seed_df, eval_df = load_benchmark_paper_results(benchmark_results)
    h2h, selfplay_summary, external = load_selfplay_results(args.selfplay_results)
    validate_manuscript_datasets(seed_df, eval_df, h2h, selfplay_summary, external)

    figure_01_workflow(args.output)
    figure_02_arena_robot_sensor_schematic(args.output)
    if not args.skip_figure3:
        if args.benchmark_root is None:
            parser.error("Fig. 3 requires --benchmark-root, or use --skip-figure3.")
        figure_03_learning_curve(args.output, args.benchmark_root)
    figure_04_seed_winrates(args.output, seed_df)
    figure_05_aggressive_winrate(args.output, seed_df)
    figure_06_win_attribution(args.output, seed_df)
    figure_07_outcome_composition(args.output, seed_df)
    figure_08_aggression(args.output, seed_df)
    figure_09_attack_contact(args.output, seed_df)
    figure_10_seed_heatmap(args.output, seed_df)
    figure_11_selfplay_official_trajectory(args.output, selfplay_summary)
    figure_12_selfplay_final_advantage(args.output, h2h)
    figure_13_selfplay_behavior(args.output, h2h)
    figure_14_external_generalization(args.output, external)

    print(f"Generated updated manuscript figures in: {args.output}")


if __name__ == "__main__":
    main()
