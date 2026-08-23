#!/usr/bin/env python3
"""
Exploratory analyses for the displacement probe, the demonstration-pace probe, and the
sampling-density probe. All sit outside the pre-registration (PROTOCOL.md 8.15, 8.16, 8.30)
and are reported separately from the registered grid.

RUN: python analysis/analyze_exploratory.py
"""

import os
import sys

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_results import newcombe_diff, wilson_ci

REG = "documents/results_full.csv"
EXP = "documents/exploratory.csv"
OUTDIR = "analysis/out_exploratory"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    reg = pd.read_csv(REG)
    exp = pd.read_csv(EXP)

    # -------- Displacement Curve -------------
    rows = []
    for seed in sorted(exp.loc[exp.condition == "clean", "seed"].unique()):
        slices = [
            ("No Displacement", "registered", reg[(reg.condition == "clean") & (reg.seed == seed) & (reg.eval_cell == "in_distribution")]),
            ("1.0 inch", "exploratory", exp[(exp.condition == "clean") & (exp.seed == seed) & (exp.eval_cell == "near_1in")]),
            ("2.0 inches", "exploratory", exp[(exp.condition == "clean") & (exp.seed == seed) & (exp.eval_cell == "near_2in")]),
            ("3.5+ inches (E1 to E5, 3.5 to 14 inches)", "registered", reg[(reg.condition == "clean") & (reg.seed == seed) & (reg.eval_cell == "new_positions")]),
        ]
        for dist, source, g in slices:
            n, k = len(g), int(g["success"].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({
                "seed": seed, "displacement": dist, "source": source, "n": n, "successes": k, "success_rate": k / n if n else float("nan"),
                "ci_low": lo, "ci_high": hi,
            })

    disp = pd.DataFrame(rows)

    probe = exp[exp.condition == "clean"]
    disp_modes = pd.crosstab([probe.eval_cell, probe.seed], probe.failure_mode)

    # ------------ Demonstration Pace ---------------

    rows = []
    for cell in ["in_distribution", "different_object"]:
        s = exp[(exp.condition == "color-slowpace") & (exp.eval_cell == cell)]
        f = reg[(reg.condition == "color") & (reg.seed == 1000) & (reg.eval_cell == cell)]
        k1, n1 = int(s["success"].sum()), len(s)
        k2, n2 = int(f["success"].sum()), len(f)
        d, lo, hi = newcombe_diff(k1, n1, k2, n2)
        _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
        rows.append({
            "eval_cell": cell,
            "slowpace": f"{k1}/{n1}",
            "slowpace_rate": k1 / n1,
            "retained_color": f"{k2}/{n2}",
            "retained_rate": k2 / n2,
            "difference": d,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "fisher_p": p,
        })

    pace = pd.DataFrame(rows)

    slow = exp[exp.condition == "color-slowpace"]
    pace_modes = pd.crosstab(slow.eval_cell, slow.failure_mode)

    # ------------ Sampling density ---------------

    rows = []
    for cell in ["in_distribution", "trained_t2", "new_positions"]:
        s = exp[(exp.condition == "density") & (exp.eval_cell == cell)]
        k1, n1 = int(s["success"].sum()), len(s)
        row = {"eval_cell": cell, "density": f"{k1}/{n1}",
               "density_rate": k1 / n1 if n1 else float("nan")}
        # Clean seed 1000 is the matched comparison wherever it ran the same cell.
        # trained_t2 has no counterpart: no other policy was evaluated at T2.
        f = reg[(reg.condition == "clean") & (reg.seed == 1000) & (reg.eval_cell == cell)]
        if len(f):
            k2, n2 = int(f["success"].sum()), len(f)
            d, lo, hi = newcombe_diff(k1, n1, k2, n2)
            _, p = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
            row.update({"clean": f"{k2}/{n2}", "clean_rate": k2 / n2,
                        "difference": d, "diff_ci_low": lo,
                        "diff_ci_high": hi, "fisher_p": p})
        rows.append(row)

    density = pd.DataFrame(rows)

    dens = exp[exp.condition == "density"]
    density_modes = pd.crosstab(dens.eval_cell, dens.failure_mode)


    # ------------ Output -----------------------
    # Round values to 4 decimal points
    disp = disp.round({"success_rate": 4, "ci_low": 4, "ci_high": 4})
    pace = pace.round({"slowpace_rate": 4, "retained_rate": 4, "difference": 4,
                    "diff_ci_low": 4, "diff_ci_high": 4, "fisher_p": 4})
    density = density.round({"density_rate": 4, "clean_rate": 4, "difference": 4,
                             "diff_ci_low": 4, "diff_ci_high": 4, "fisher_p": 4})

    disp.to_csv(os.path.join(OUTDIR, "displacement_curve.csv"), index=False)
    pace.to_csv(os.path.join(OUTDIR, "pace_comparison.csv"), index=False)
    density.to_csv(os.path.join(OUTDIR, "density_probe.csv"), index=False)

    disp_modes.to_csv(os.path.join(OUTDIR, "displacement_failure_modes.csv"))
    pace_modes.to_csv(os.path.join(OUTDIR, "pace_failure_modes.csv"))
    density_modes.to_csv(os.path.join(OUTDIR, "density_failure_modes.csv"))

    print("--- Displacement Probe: Clean Policy vs. Distance from T6 ---")
    print(disp.to_string(index=False))
    print("\n--- Displacement Probe Failure Modes ---")
    print(disp_modes.to_string())
    print("\n--- Demonstration Pace: Slowpace vs. Retained Color, Seed 1000 ---")
    print(pace.to_string(index=False))
    print("\n--- Pace Probe Failure Modes ---")
    print(pace_modes.to_string())
    print("\n--- Density: Two positions vs. Clean, Seed 1000 ---")
    print(density.to_string(index=False))
    print("\n--- Density Probe Failure Modes ---")
    print(density_modes.to_string())
    print(f"\nSaved to {OUTDIR}/")

if __name__ == "__main__":
    main()