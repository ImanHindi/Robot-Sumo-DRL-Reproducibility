from __future__ import annotations

from pathlib import Path


def _safe_imports():
    import matplotlib.pyplot as plt
    import pandas as pd
    return plt, pd


def make_training_plots(csv_path: str | Path, out_dir: str | Path) -> None:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    plt, pd = _safe_imports()
    df = pd.read_csv(path)
    if df.empty:
        return
    if "valid_episode" in df.columns:
        valid = df["valid_episode"].fillna(1).astype(int) == 1
        df = df.loc[valid].copy()
        if df.empty:
            return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = df.get("total_env_steps", df.index.to_series())
    window = max(5, min(50, len(df) // 10 if len(df) >= 10 else 5))

    for col, title, ylabel, fname in [
        ("episode_reward", "Training reward", "Episode reward", "training_reward"),
        ("win", "Rolling win rate", "Win rate", "training_win_rate"),
        ("aggression_index", "Aggression index", "Aggression index", "training_aggression"),
        ("target_detected_ratio", "Target detection ratio", "Fraction of episode", "training_target_detection"),
        ("attack_ratio", "Translational attack ratio", "Fraction of episode", "training_attack_ratio"),
        ("contact_ratio", "Physical contact ratio", "Fraction of episode", "training_contact_ratio"),
        ("contact_forward_ratio", "Translational pressure while in contact", "Fraction of episode", "training_contact_forward_ratio"),
        ("translational_forward_ratio", "Forward-translational motion ratio", "Fraction of episode", "training_translational_forward_ratio"),
        ("straight_forward_ratio", "Straight-forward motion ratio", "Fraction of episode", "training_straight_forward_ratio"),
        ("curved_forward_ratio", "Curved-forward motion ratio", "Fraction of episode", "training_curved_forward_ratio"),
        ("front_visible_nonforward_ratio", "Visible-target non-attack ratio", "Fraction of episode", "training_nonattack_ratio"),
        ("search_forward_ratio", "Safe search-forward ratio", "Fraction of episode", "training_search_ratio"),
        ("episode_steps", "Episode duration", "Simulation steps", "training_episode_length"),
    ]:
        if col not in df.columns:
            continue
        y = df[col].rolling(window, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(x, y, linewidth=1.7)
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"{fname}.png", dpi=300)
        fig.savefig(out / f"{fname}.pdf")
        plt.close(fig)


def make_evaluation_plots(csv_path: str | Path, out_dir: str | Path) -> None:
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    plt, pd = _safe_imports()
    df = pd.read_csv(path)
    if df.empty:
        return
    if "valid_episode" in df.columns:
        valid = df["valid_episode"].fillna(1).astype(int) == 1
        df = df.loc[valid].copy()
        if df.empty:
            return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if {"win", "loss", "draw"}.issubset(df.columns):
        vals = [df["win"].mean(), df["loss"].mean(), df["draw"].mean()]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(["Win", "Loss", "Draw"], vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of matches")
        ax.set_title("Evaluation outcomes")
        fig.tight_layout()
        fig.savefig(out / "evaluation_outcomes.png", dpi=300)
        fig.savefig(out / "evaluation_outcomes.pdf")
        plt.close(fig)

    if "aggression_index" in df.columns:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.hist(df["aggression_index"].dropna(), bins=min(20, max(5, len(df) // 5)))
        ax.set_xlabel("Aggression index")
        ax.set_ylabel("Episodes")
        ax.set_title("Evaluation aggression distribution")
        fig.tight_layout()
        fig.savefig(out / "evaluation_aggression.png", dpi=300)
        fig.savefig(out / "evaluation_aggression.pdf")
        plt.close(fig)

    behavior_cols = [c for c in ["target_detected_ratio", "engagement_ratio", "attack_ratio",
                                  "translational_forward_ratio", "straight_forward_ratio", "curved_forward_ratio",
                                  "contact_ratio", "contact_forward_ratio",
                                  "front_visible_nonforward_ratio", "search_forward_ratio",
                                  "scan_ratio", "edge_risk_ratio"] if c in df.columns]
    if behavior_cols:
        labels = [c.replace("_ratio", "").replace("_", " ").title() for c in behavior_cols]
        vals = [float(df[c].mean()) for c in behavior_cols]
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.bar(labels, vals)
        ax.set_ylim(0, max(0.05, min(1.0, max(vals) * 1.25 if vals else 1.0)))
        ax.set_ylabel("Mean fraction of episode")
        ax.set_title("Evaluation behavioral profile")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(out / "evaluation_behavior_profile.png", dpi=300)
        fig.savefig(out / "evaluation_behavior_profile.pdf")
        plt.close(fig)
