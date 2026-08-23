#!/usr/bin/env python3
"""
analysis/seed_variance.py

How much does a condition's success rate move between training seeds? PROTOCOL.md §4.7
forbids comparing conditions across seeds, so analyze_results.py refuses multi-seed input.
This script does the one comparison that is legitimate: the same condition against itself,
seed 1000 versus seed 2000, which bounds what a single-seed version of this study could
have concluded.

The calculated difference is seed 2000 - seed 1000, so a negative value indicates that
seed 2000 performed worse than the seed 1000.

Intervals here treat episodes as independent, but they are not: episodes within a cell share a
scene. These intervals are therefore narrower than the cluster-corrected ones used for the
registered comparisons, and this script is a bound on seed sensitivity rather than a
reportable test.

RUN: python analysis/seed_variance.py
"""

import os

import pandas as pd
from scipy.stats import fisher_exact

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import newcombe_diff, wilson_ci

REG = "documents/results_full.csv"
OUTDIR = "analysis/out_seed_variance"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    d = pd.read_csv(REG)
    rows = []

    for cond, g in d.groupby("condition"):
        a, b = g[g.seed == 1000].success, g[g.seed == 2000].success
        ka, na, kb, nb = int(a.sum()), len(a), int(b.sum()), len(b)
        diff, lo, hi = newcombe_diff(kb, nb, ka, na)
        _, p = fisher_exact([[kb, nb - kb], [ka, na - ka]])
        l1, h1 = wilson_ci(ka, na)
        l2, h2 = wilson_ci(kb, nb)
        rows.append({
            "condition": cond,
            "seed1000": f"{ka}/{na}",
            "rate1000": ka / na,
            "ci1000_low": l1,
            "ci1000_high": h1,
            "seed2000": f"{kb}/{nb}",
            "rate2000": kb / nb,
            "ci2000_low": l2,
            "ci2000_high": h2,
            "difference": diff,
            "diff_ci_low": lo,
            "diff_ci_high": hi,
            "fisher_p": p,
        })

    ka, na = int(d[d.seed == 1000].success.sum()), int((d.seed == 1000).sum())
    kb, nb = int(d[d.seed == 2000].success.sum()), int((d.seed == 2000).sum())
    diff, lo, hi = newcombe_diff(kb, nb, ka, na)
    _, p = fisher_exact([[kb, nb - kb], [ka, na - ka]])
    rows.append({
        "condition": "ALL",
        "seed1000": f"{ka}/{na}",
        "rate1000": ka / na,
        "seed2000": f"{kb}/{nb}",
        "rate2000": kb / nb,
        "difference": diff,
        "diff_ci_low": lo,
        "diff_ci_high": hi,
        "fisher_p": p
    })

    t = pd.DataFrame(rows).round(4)
    t.to_csv(os.path.join(OUTDIR, "seed_variance.csv"), index=False)
    print(t.to_string(index=False))

    w = t[t.condition != "ALL"]
    per = int(d[d.seed == 1000].groupby("condition").size().iloc[0])
    print(f"\nlargest within-condition swing: {w.difference.abs().max():.3f} "
          f"({w.loc[w.difference.abs().idxmax(), 'condition']}), "
          f"on {per} rollouts per condition per seed")

    print(f"\nsaved to {OUTDIR}/")

if __name__ == "__main__":
    main()