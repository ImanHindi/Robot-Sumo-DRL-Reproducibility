from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BEHAVIOR_METRICS = [
    "aggression_index", "attack_ratio", "contact_ratio", "contact_forward_ratio",
    "push_outward_progress_ratio", "edge_risk_ratio", "translational_forward_ratio",
]


def text_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "(no rows)"
    return "```text\n" + df.to_string(index=False) + "\n```"


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "valid_episode" in df.columns:
        df = df[pd.to_numeric(df["valid_episode"], errors="coerce").fillna(1).astype(int) == 1].copy()
    return df


def mean_metric(df: pd.DataFrame, name: str) -> float:
    if name not in df.columns or len(df) == 0:
        return math.nan
    return float(pd.to_numeric(df[name], errors="coerce").mean())


def summarize_actor(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {}
    out = {
        "episodes": int(len(df)),
        "win_rate": mean_metric(df, "win"),
        "loss_rate": mean_metric(df, "loss"),
        "draw_rate": mean_metric(df, "draw"),
        "forced_win_rate": mean_metric(df, "forced_win"),
        "unforced_win_rate": mean_metric(df, "unforced_win"),
        "mean_reward": mean_metric(df, "episode_reward"),
    }
    for m in BEHAVIOR_METRICS:
        out[m] = mean_metric(df, m)
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 20260914) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2 or np.allclose(a[mask], b[mask]):
        return math.nan, 1.0 if mask.sum() else math.nan
    try:
        result = stats.wilcoxon(a[mask], b[mask], alternative="two-sided", method="auto")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return math.nan, math.nan


def parse_step(label: str) -> int | None:
    m = re.search(r"step_(\d+)_", label)
    return int(m.group(1)) if m else None


def head_to_head_rows(root: Path) -> pd.DataFrame:
    rows = []
    pairs_root = root / "pairs"
    if not pairs_root.exists():
        return pd.DataFrame()
    for pair in sorted(pairs_root.glob("seed_*")):
        try:
            seed = int(pair.name.split("_")[-1])
        except Exception:
            continue
        hroot = pair / "head_to_head"
        if not hroot.exists():
            continue
        labels = sorted({
            p.name.replace("_veteran_as_agent0", "").replace("_novice_as_agent0", "")
            for p in hroot.iterdir() if p.is_dir() and p.name.startswith("step_")
        })
        for base in labels:
            step = parse_step(base + "_")
            if step is None:
                continue
            pa = hroot / f"{base}_veteran_as_agent0" / "agent_0" / "evaluation.csv"
            pb = hroot / f"{base}_novice_as_agent0" / "agent_0" / "evaluation.csv"
            a = read_csv(pa); b = read_csv(pb)
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                continue
            total = len(a) + len(b)
            veteran_wins = float(pd.to_numeric(a["win"], errors="coerce").fillna(0).sum() + pd.to_numeric(b["loss"], errors="coerce").fillna(0).sum())
            novice_wins = float(pd.to_numeric(b["win"], errors="coerce").fillna(0).sum() + pd.to_numeric(a["loss"], errors="coerce").fillna(0).sum())
            draws = float(pd.to_numeric(a["draw"], errors="coerce").fillna(0).sum() + pd.to_numeric(b["draw"], errors="coerce").fillna(0).sum())

            forced_veteran = float(
                pd.to_numeric(a.get("forced_win", 0), errors="coerce").fillna(0).sum()
                + (((pd.to_numeric(b["loss"], errors="coerce").fillna(0) == 1) & (b.get("ring_out_cause", "") == "contact_assisted")).sum())
            )
            forced_novice = float(
                pd.to_numeric(b.get("forced_win", 0), errors="coerce").fillna(0).sum()
                + (((pd.to_numeric(a["loss"], errors="coerce").fillna(0) == 1) & (a.get("ring_out_cause", "") == "contact_assisted")).sum())
            )
            row = {
                "seed": seed,
                "stage3_steps": step,
                "episodes_total_role_swapped": total,
                "veteran_win_rate": veteran_wins / total,
                "novice_win_rate": novice_wins / total,
                "draw_rate": draws / total,
                "veteran_forced_win_rate": forced_veteran / total,
                "novice_forced_win_rate": forced_novice / total,
                "novice_advantage": (novice_wins - veteran_wins) / total,
            }
            # Behavioral metrics come from the agent when it occupies agent 0.
            for m in BEHAVIOR_METRICS:
                row[f"veteran_{m}"] = mean_metric(a, m)
                row[f"novice_{m}"] = mean_metric(b, m)
            rows.append(row)
    return pd.DataFrame(rows)


def training_rows(root: Path, bin_steps: int = 50000) -> pd.DataFrame:
    rows = []
    for pair in sorted((root / "pairs").glob("seed_*")) if (root / "pairs").exists() else []:
        try:
            seed = int(pair.name.split("_")[-1])
        except Exception:
            continue
        for agent_id, role in ((0, "veteran"), (1, "novice")):
            path = pair / "stage3_selfplay" / f"agent_{agent_id}" / f"train_episodes_agent{agent_id}.csv"
            df = read_csv(path)
            if df is None or len(df) == 0 or "total_env_steps" not in df.columns:
                continue
            df = df.copy()
            df["total_env_steps"] = pd.to_numeric(df["total_env_steps"], errors="coerce")
            df = df.dropna(subset=["total_env_steps"])
            df["bin_steps"] = ((df["total_env_steps"] - 1) // bin_steps + 1) * bin_steps
            for b, g in df.groupby("bin_steps"):
                row = {
                    "seed": seed, "role": role, "stage3_steps": int(b), "episodes": len(g),
                    "win_rate": mean_metric(g, "win"), "loss_rate": mean_metric(g, "loss"),
                    "draw_rate": mean_metric(g, "draw"), "forced_win_rate": mean_metric(g, "forced_win"),
                }
                for m in BEHAVIOR_METRICS:
                    row[m] = mean_metric(g, m)
                rows.append(row)
    return pd.DataFrame(rows)


def external_rows(root: Path, benchmark_root: Path) -> pd.DataFrame:
    rows = []
    seeds = sorted({int(p.name.split("_")[-1]) for p in (root / "pairs").glob("seed_*")}) if (root / "pairs").exists() else []
    for seed in seeds:
        pair = root / "pairs" / f"seed_{seed}"
        for opp in ("static", "scripted", "aggressive"):
            pre = benchmark_root / "PPO" / f"seed_{seed}" / f"eval_{opp}" / "agent_0" / "evaluation.csv"
            for role, path in (
                ("veteran_pre", pre),
                ("veteran_post", pair / "external" / "veteran_post" / opp / "agent_0" / "evaluation.csv"),
                ("novice_post", pair / "external" / "novice_post" / opp / "agent_0" / "evaluation.csv"),
            ):
                df = read_csv(path)
                if df is None or len(df) == 0:
                    continue
                row = {"seed": seed, "role": role, "opponent": opp, **summarize_actor(df)}
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_h2h(h2h: pd.DataFrame) -> pd.DataFrame:
    if h2h.empty:
        return pd.DataFrame()
    rows = []
    for step, g in h2h.groupby("stage3_steps"):
        row = {"stage3_steps": int(step), "n_seeds": int(g.seed.nunique())}
        for metric in ["veteran_win_rate", "novice_win_rate", "draw_rate", "veteran_forced_win_rate", "novice_forced_win_rate", "novice_advantage"]:
            vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(float)
            lo, hi = bootstrap_ci(vals)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_sd"] = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else math.nan
            row[f"{metric}_ci95_low"] = lo; row[f"{metric}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stage3_steps")


def statistical_tests(h2h: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not h2h.empty:
        final_step = h2h.stage3_steps.max()
        f = h2h[h2h.stage3_steps == final_step].sort_values("seed")
        w, p = paired_wilcoxon(f.novice_win_rate.to_numpy(), f.veteran_win_rate.to_numpy())
        diffs = f.novice_win_rate.to_numpy(float) - f.veteran_win_rate.to_numpy(float)
        lo, hi = bootstrap_ci(diffs)
        rows.append({
            "comparison": "final_h2h_novice_vs_veteran_win_rate",
            "stage3_steps": int(final_step), "n_pairs": len(f), "test": "paired Wilcoxon signed-rank",
            "statistic": w, "p_value": p, "mean_paired_difference": float(np.mean(diffs)),
            "median_paired_difference": float(np.median(diffs)), "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
            "note": "Seed is the independent unit; with n=5, inferential power is intentionally limited.",
        })
    if not external.empty:
        for opp in ("static", "scripted", "aggressive"):
            piv = external[external.opponent == opp].pivot(index="seed", columns="role", values="forced_win_rate")
            if {"veteran_pre", "veteran_post"}.issubset(piv.columns):
                a = piv["veteran_post"].to_numpy(float); b = piv["veteran_pre"].to_numpy(float)
                w, p = paired_wilcoxon(a, b); d = a - b; lo, hi = bootstrap_ci(d)
                rows.append({
                    "comparison": f"veteran_forced_win_post_minus_pre_{opp}", "stage3_steps": math.nan,
                    "n_pairs": len(d), "test": "paired Wilcoxon signed-rank", "statistic": w, "p_value": p,
                    "mean_paired_difference": float(np.mean(d)), "median_paired_difference": float(np.median(d)),
                    "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
                    "note": "Uses the same benchmark evaluation seed before and after self-play.",
                })
    return pd.DataFrame(rows)


def parity_table(h2h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if h2h.empty:
        return pd.DataFrame()
    for seed, g in h2h.sort_values("stage3_steps").groupby("seed"):
        reached = g[g.novice_win_rate >= g.veteran_win_rate]
        first = int(reached.stage3_steps.iloc[0]) if len(reached) else math.nan
        final = g.iloc[-1]
        rows.append({
            "seed": int(seed), "first_checkpoint_novice_parity_steps": first,
            "final_veteran_win_rate": final.veteran_win_rate,
            "final_novice_win_rate": final.novice_win_rate,
            "final_novice_advantage": final.novice_advantage,
            "final_veteran_forced_win_rate": final.veteran_forced_win_rate,
            "final_novice_forced_win_rate": final.novice_forced_win_rate,
        })
    return pd.DataFrame(rows)


def make_figures(out: Path, h2h: pd.DataFrame, summary: pd.DataFrame, training: pd.DataFrame, external: pd.DataFrame) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        x = summary.stage3_steps.to_numpy(float) / 1000.0
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for prefix, label in (("veteran", "Veteran"), ("novice", "Novice")):
            y = summary[f"{prefix}_win_rate_mean"].to_numpy(float)
            lo = summary[f"{prefix}_win_rate_ci95_low"].to_numpy(float)
            hi = summary[f"{prefix}_win_rate_ci95_high"].to_numpy(float)
            ax.plot(x, y, marker="o", label=label)
            ax.fill_between(x, lo, hi, alpha=0.18)
        ax.set_xlabel("Stage-3 self-play steps (thousands)"); ax.set_ylabel("Role-swapped deterministic win rate")
        ax.set_ylim(bottom=0); ax.set_title("Veteran–novice head-to-head adaptation"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "selfplay_h2h_trajectory.png", dpi=300); fig.savefig(out / "selfplay_h2h_trajectory.pdf"); plt.close(fig)

    if not h2h.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for seed, g in h2h.sort_values("stage3_steps").groupby("seed"):
            ax.plot(g.stage3_steps / 1000.0, g.novice_advantage, marker="o", label=f"Seed {seed}")
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Stage-3 self-play steps (thousands)"); ax.set_ylabel("Novice win rate − veteran win rate")
        ax.set_title("Seed-specific adaptation and parity crossing"); ax.legend(ncol=2); fig.tight_layout()
        fig.savefig(out / "selfplay_seed_advantage.png", dpi=300); fig.savefig(out / "selfplay_seed_advantage.pdf"); plt.close(fig)

        for metric, title in (("aggression_index", "Aggression index"), ("contact_ratio", "Contact ratio"), ("attack_ratio", "Attack ratio"), ("edge_risk_ratio", "Edge-risk ratio")):
            fig, ax = plt.subplots(figsize=(8.0, 4.6))
            agg = h2h.groupby("stage3_steps")[[f"veteran_{metric}", f"novice_{metric}"]].mean().reset_index()
            ax.plot(agg.stage3_steps / 1000.0, agg[f"veteran_{metric}"], marker="o", label="Veteran")
            ax.plot(agg.stage3_steps / 1000.0, agg[f"novice_{metric}"], marker="o", label="Novice")
            ax.set_xlabel("Stage-3 self-play steps (thousands)"); ax.set_ylabel(title); ax.set_title(f"Self-play behavioral trajectory: {title}"); ax.legend(); fig.tight_layout()
            stem = metric.replace("_ratio", "")
            fig.savefig(out / f"selfplay_{stem}_trajectory.png", dpi=300); fig.savefig(out / f"selfplay_{stem}_trajectory.pdf"); plt.close(fig)

    if not training.empty:
        agg_t = training.groupby(["stage3_steps", "role"], as_index=False).agg(
            forced_win_rate=("forced_win_rate", "mean"),
            win_rate=("win_rate", "mean"),
            aggression_index=("aggression_index", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for role, label in (("veteran", "Veteran"), ("novice", "Novice")):
            g = agg_t[agg_t.role == role].sort_values("stage3_steps")
            if len(g):
                ax.plot(g.stage3_steps / 1000.0, g.forced_win_rate, marker="o", label=label)
        ax.set_xlabel("Stage-3 self-play steps (thousands)"); ax.set_ylabel("Mean forced-win rate in stochastic training episodes")
        ax.set_ylim(bottom=0); ax.set_title("Asymmetric self-play training trajectory"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "selfplay_training_forced_win_trajectory.png", dpi=300); fig.savefig(out / "selfplay_training_forced_win_trajectory.pdf"); plt.close(fig)

    if not external.empty:
        agg = external.groupby(["opponent", "role"], as_index=False).forced_win_rate.mean()
        roles = [r for r in ["veteran_pre", "veteran_post", "novice_post"] if r in set(agg.role)]
        opponents = [o for o in ["static", "scripted", "aggressive"] if o in set(agg.opponent)]
        x = np.arange(len(opponents), dtype=float); width = 0.8 / max(1, len(roles))
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for i, role in enumerate(roles):
            vals = [float(agg[(agg.opponent == o) & (agg.role == role)].forced_win_rate.iloc[0]) if len(agg[(agg.opponent == o) & (agg.role == role)]) else math.nan for o in opponents]
            ax.bar(x + (i - (len(roles)-1)/2)*width, vals, width=width, label=role.replace("_", " ").title())
        ax.set_xticks(x, [o.title() for o in opponents]); ax.set_ylabel("Forced-win rate"); ax.set_ylim(bottom=0)
        ax.set_title("External generalization before and after asymmetric self-play"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "selfplay_external_generalization.png", dpi=300); fig.savefig(out / "selfplay_external_generalization.pdf"); plt.close(fig)


def write_interpretation(out: Path, h2h: pd.DataFrame, parity: pd.DataFrame, external: pd.DataFrame, tests: pd.DataFrame) -> None:
    lines = [
        "# Asymmetric Veteran-versus-Novice Self-Play: Automated Scientific Summary",
        "",
        "This file is generated from completed experiment outputs. Treat seed as the independent training replicate.",
        "The head-to-head evaluation is deterministic and role-swapped to reduce robot-slot/spawn bias.",
        "",
    ]
    if h2h.empty:
        lines += ["No complete head-to-head checkpoint results are available yet.", ""]
    else:
        final_step = int(h2h.stage3_steps.max()); f = h2h[h2h.stage3_steps == final_step]
        lines += [
            f"## Final checkpoint ({final_step:,} Stage-3 steps)",
            f"- Mean veteran win rate: {f.veteran_win_rate.mean():.3f}",
            f"- Mean novice win rate: {f.novice_win_rate.mean():.3f}",
            f"- Mean draw rate: {f.draw_rate.mean():.3f}",
            f"- Mean novice-minus-veteran win-rate difference: {f.novice_advantage.mean():+.3f}",
            f"- Mean veteran forced-win rate: {f.veteran_forced_win_rate.mean():.3f}",
            f"- Mean novice forced-win rate: {f.novice_forced_win_rate.mean():.3f}",
            "",
            "Interpretation rule: a novice advantage demonstrates adaptation only if it is reproduced across multiple seeds and is not accompanied solely by opponent self-outs. Forced-win and contact metrics should be inspected alongside official win rate.",
            "",
        ]
    if not parity.empty:
        lines += ["## Parity by seed", text_table(parity), ""]
    if not tests.empty:
        lines += ["## Paired statistical tests", text_table(tests), ""]
    if not external.empty:
        e = external.groupby(["opponent", "role"], as_index=False).forced_win_rate.mean()
        lines += [
            "## External generalization",
            text_table(e),
            "",
            "External performance is essential: a novice that beats only its co-adapting veteran may have learned an opponent-specific exploit rather than a generally stronger sumo policy. Veteran pre/post comparison also reveals possible catastrophic forgetting.",
            "",
        ]
    (out / "SELFPLAY_SCIENTIFIC_INTERPRETATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--benchmark-root", required=True)
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve(); benchmark_root = Path(args.benchmark_root).expanduser().resolve()
    out = root / "paper_results"; figs = out / "figures"; out.mkdir(parents=True, exist_ok=True)

    h2h = head_to_head_rows(root); training = training_rows(root); external = external_rows(root, benchmark_root)
    summary = summarize_h2h(h2h); parity = parity_table(h2h); tests = statistical_tests(h2h, external)

    if not h2h.empty: h2h.to_csv(out / "selfplay_head_to_head_per_seed_checkpoint.csv", index=False)
    if not summary.empty: summary.to_csv(out / "selfplay_head_to_head_summary_ci95.csv", index=False)
    if not training.empty: training.to_csv(out / "selfplay_training_trajectory.csv", index=False)
    if not external.empty: external.to_csv(out / "selfplay_external_generalization_per_seed.csv", index=False)
    if not parity.empty: parity.to_csv(out / "selfplay_parity_by_seed.csv", index=False)
    if not tests.empty: tests.to_csv(out / "selfplay_statistical_tests.csv", index=False)

    make_figures(figs, h2h, summary, training, external)
    write_interpretation(out, h2h, parity, external, tests)
    print(f"Asymmetric self-play analysis written to: {out}")


if __name__ == "__main__":
    main()
