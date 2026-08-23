#!/usr/bin/env python3
""" 
Turns raw eval cell results (one success / fail label per episode) into the paper's tables and plots

Input is a results CSV file, each row is an episode, and the columns are: condition, eval_cell, seed, episode, success

The two PNGs written here are diagnostics for reading the CSVs at a glance; the paper's figures are built separately by analysis/make_figures.py

RUN, once per seed (PROTOCOL.md §4.7 forbids pooling):
    python analysis/analyze_results.py documents/results_seed1000.csv --outdir analysis/out_seed1000
    python analysis/analyze_results.py documents/results_seed2000.csv --outdir analysis/out_seed2000
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, norm

CONF = 0.95 # 95% Confidence Intervals
Z95 = norm.ppf(1 - (1 - CONF) / 2) # Z-Score

def wilson_ci(successes, n, z=Z95):
    # 95% Wilson score interval for a success rate, it's reliable at a small N / extreme rates
    # Unlike textbook p +/- 1.96*sqrt(p(1-p)/n), z = 1.96 for 95% confidence
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, (center - half)), min(1.0, (center + half))

def newcombe_diff(k1, n1, k2, n2, z=Z95):
    # Newcombe hybrid score interval for p1 - p2
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lower = d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
    upper = d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
    return d, max(-1.0, lower), min(1.0, upper)

def load_results(path):
    # Reads the CSV and sanity checks it so a typo doesn't corrupt the analysis, checks columns names and success values
    df = pd.read_csv(path)
    required = {"condition", "eval_cell", "seed", "episode", "success"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing the following columns: {missing}")
    df["success"] = df["success"].astype(int)
    bad = set(df["success"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"'success' includes: {bad}; 'success' must only be either 0 or 1")
    return df

def summarize(df):
    # Success rate and the Wilson CI per (condition, eval_cell)
    # Groupby splits the table into one sub-table per combination, and these are summarized to a single row
    rows = []
    for (cond, cell), g in df.groupby(["condition", "eval_cell"]):
        n = len(g)
        s = int(g["success"].sum())
        lo, hi = wilson_ci(s, n)
        rows.append({
            "condition": cond, "eval_cell": cell,
            "n": n, "successes": s, 
            "success_rate": s / n if n else np.nan,
            "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows).sort_values(["condition", "eval_cell"]).reset_index(drop=True)

def generalization_gap(summary, in_dist_cell="in_distribution"):
    # Per condition: in-distribution rate minus held-out rate, smaller means it generalizes better
    # Unweighted mean of the four held-out cell rates. Equal to the pooled rate
    # because every cell has the same n; revisit if n ever changes.

    rows = []
    for cond, g in summary.groupby("condition"):
        in_dist = g.loc[g["eval_cell"] == in_dist_cell, "success_rate"]
        held = g.loc[g["eval_cell"] != in_dist_cell, "success_rate"]
        in_rate = float(in_dist.iloc[0]) if len(in_dist) else np.nan
        held_rate = float(held.mean()) if len(held) else np.nan
        rows.append({
            "condition": cond,
            "in_distribution": in_rate,
            "held_out_mean": held_rate,
            "generalization_gap": in_rate - held_rate,
        })
    return pd.DataFrame(rows).sort_values("generalization_gap").reset_index(drop=True)

def matched_comparisons(summary, pairs, baseline="clean"):
    # For each pair (cell to matched condition), compare the matched conditions rate vs. baseline (clean)
    # on that cell. Reports the difference in success rate with a Newcombe hybrid score interval on the
    # difference and a fisher exact test
    
    def counts(cond, cell):
        row = summary[(summary["condition"] == cond) & (summary["eval_cell"] == cell)]
        if row.empty:
            return None
        r = row.iloc[0]
        return int(r["successes"]), int(r["n"])

    rows = []
    for cell, matched_cond in pairs.items():
        m = counts(matched_cond, cell)
        b = counts(baseline, cell)
        if m is None or b is None:
            continue
        k1, n1 = m
        k2, n2 = b
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        _, p = fisher_exact([[k1, n1-k1], [k2, n2-k2]])
        rows.append({
            "eval_cell": cell,
            "matched_condition": matched_cond,
            "matched": f"{k1}/{n1}",
            "matched_rate": k1 / n1,
            "baseline": baseline, 
            "baseline_counts": f"{k2}/{n2}",
            "baseline_rate": k2 / n2,
            "difference": d,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "fisher_p": p,
        })
    return pd.DataFrame(rows)

def plot_success(summary, path):
    conditions = sorted(summary["condition"].unique())
    cells = sorted(summary["eval_cell"].unique())
    x = np.arange(len(cells))
    w = 0.8 / max(len(conditions), 1)
    fig, ax = plt.subplots(figsize=(9,5))
    for i, cond in enumerate(conditions):
        g = summary[summary["condition"]==cond].set_index("eval_cell").reindex(cells)
        rates = g["success_rate"].values
        err = np.clip([rates - g["ci_low"].values, g["ci_high"].values - rates], 0, None)
        ax.bar(x + i * w, rates, w, yerr = err, capsize=3, label=cond)
    ax.set_xticks(x + w * (len(conditions) - 1) /2)
    ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success rate by condition and eval cell (Wilson 95% CI)")
    ax.legend(title = "Training condition")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_gap(gap, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(gap["condition"], gap["generalization_gap"])
    ax.set_ylabel("Generalization gap \n(in-dist - held-out)")
    ax.set_title("Generalization gap by condition (lower means generalizes better)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="path to results.csv")
    parser.add_argument("--outdir", default="analysis/out")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_results(args.results)
    seeds = sorted(df["seed"].unique())
    if len(seeds) > 1:
        raise SystemExit(
            f"{args.results} contains seeds {seeds}. PROTOCOL.md §4.7 and §8.14 forbid\n"
            f"pooling seeds. Run once per seed:\n"
            f" python analysis/analyze_results.py documents/results_seed1000.csv "
            f"--outdir analysis/out_seed1000\n"
            f" python analysis/analyze_results.py documents/results_seed2000.csv "
            f"--outdir analysis/out_seed2000"
        )
    print(f"seed {seeds[0]}, {len(df)} episodes\n")

    summary = summarize(df)
    gap = generalization_gap(summary)
    pairs = {"new_positions": "randomized", "different_object": "color"}
    matched = matched_comparisons(summary, pairs)

    summary.insert(0, "seed", seeds[0])
    gap.insert(0, "seed", seeds[0])
    if not matched.empty:
        matched.insert(0, "seed", seeds[0])


    summary.to_csv(os.path.join(args.outdir, "success_by_cell.csv"), index=False)
    gap.to_csv(os.path.join(args.outdir, "generalization_gap.csv"), index=False)
    matched.to_csv(os.path.join(args.outdir, "matched_comparisons.csv"), index=False)
    plot_success(summary, os.path.join(args.outdir, "success_by_cell.png"))
    plot_gap(gap, os.path.join(args.outdir, "generalization_gap.png"))

    print("=== Success rate by condition x eval cell ===")
    print(summary.to_string(index=False))
    print("\n=== Generalization gap by condition (best to worst) ===")
    print(gap.to_string(index=False))
    print("\n=== Matched-axis comparisons (matched condition vs clean) ===")
    print(matched.to_string(index=False))
    print(f"\nSaved tables and plots to {args.outdir}/")

if __name__ == "__main__":
    main()