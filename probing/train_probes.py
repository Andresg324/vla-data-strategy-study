"""
probing/train_probes.py

Trains linear probes to predict episode success from a policy's hidden activations, one
probe per condition and training seed.

Evaluation design follows prior work by the author on linear probes in language models
(citation withheld for review), which is why it reports AUROC only and uses grouped splits.

Four choices follow from it:

  - AUROC only. Accuracy depends on a decision threshold, so an intervention that shifts the
    boundary without damaging the representation reads as a large effect. In that paper an
    ablation dropped probe accuracy by 0.23 while AUROC moved 0.01, and three such
    reversals appeared before the pattern was understood.
  - Episode-grouped splits. Timesteps within an episode are not independent, so an ungrouped
    split trains and tests on the same episode.
  - Repeated splits with a percentile interval, since one split at this sample size reports
    a single draw from a wide distribution.
  - Within-cell versus pooled AUROC. Evaluation cells differ in difficulty, so a probe can
    score well by learning which cell it is in rather than anything about the policy's state.
    A large pooled-minus-within-cell gap is that confound surfacing.

Not reportable on this study's data: see analysis/README.md. Kept for provenance; the
outcome analysis in the paper comes from probe_success.py.

RUN: python probing/train_probes.py probing/out_np/activations_clean.npz
"""

import argparse, os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore", message=".*matmul.*")

def grouped_auroc(X, y, groups, seed):
    # One episode-grouped split to test AUROC (nan if a split is degenerate).
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    # StandardScaler on near-constant activation dimensions raises FPU flags. Inputs are
    # finite and the scores are asserted finite below, so the flags are noise. Scoped rather
    # than global, so an overflow anywhere else still surfaces.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        clf.fit(X[tr], y[tr])
        scores = clf.predict_proba(X[te])[:, 1]
    assert np.isfinite(scores).all(), "probe produced non-finite scores"
    return roc_auc_score(y[te], scores)

def auroc_with_ci(X, y, groups, n_repeats=100):
    # Median and 2.5 / 97.5 percentiles over repeated grouped splits, plus how many
    # splits were usable. A CI built from a handful of surviving splits is not a CI.
    vals = [grouped_auroc(X, y, groups, s) for s in range(n_repeats)]
    vals = [v for v in vals if v == v]
    if not vals:
        return np.nan, np.nan, np.nan, 0
    return(float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals))

def within_cell_auroc(X, y, cells, groups, n_repeats=40):
    # De-confound: AUROC inside each cell (control for difficulty), and then averaged
    per = []
    for cl in np.unique(cells):
        m = cells == cl
        if len(set(y[m])) < 2:
            continue
        med, _, _, _ = auroc_with_ci(X[m], y[m], groups[m], n_repeats)
        if med == med:
            per.append(med)
    return float(np.mean(per)) if per else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("activations"); ap.add_argument("--outdir", default="probing/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.activations, allow_pickle=True)
    X, condition, eval_cell = d["X"], d["condition"], d["eval_cell"]
    episode, success, tfe = d["episode"], d["success"].astype(int), d["t_from_end"]
    if len(set(success.tolist())) < 2:
        raise SystemExit(
            f"{args.activations} contains only success={sorted(set(success.tolist()))}. "
            "A success probe needs both outcomes; the New Positions cell is 0/15 for every "
            "policy. Extract activations for a cell with mixed outcomes first."
        )

    print(f"{'condition':12s} {'seed':>5s} {'pooled AUROC [95% CI]':30s} {'within-cell':12s} gap")
    results = []
    seed_arr = d["seed"]
    for c in sorted(set(condition)):
        for s in sorted(set(seed_arr[condition == c])):
            m = (condition == c) & (seed_arr == s)
            med, lo, hi, n_valid = auroc_with_ci(X[m], success[m], episode[m])
            wc = within_cell_auroc(X[m], success[m], eval_cell[m], episode[m])
            gap = med - wc if (med == med and wc == wc) else np.nan
            results.append((c, int(s), med, lo, hi, wc, gap, n_valid))
            ci = f"{med:.3f} [{lo:.3f}, {hi:.3f}]"
            print(f"{c:12s} {s:>5d} {ci:30s} {wc:<12.3f}  {gap:+.3f} n={n_valid}")

    # Lead time to decodable failure (pooled across conditions and seeds, descriptive)
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, success, episode))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], success[tr])
    p, yt, bt = clf.predict_proba(X[te])[:, 1], success[te], tfe[te]

    # Bin t_from_end instead of using exact values. Activations are captured at
    # SmolVLA's action-chunk boundaries, so the raw t_from_end values are sparse
    # and irregular. Grouping by exact value leaves one or two samples per value,
    # nearly all single-class, and the curve collapses to a few points.
    
    BIN = 10 # Steps per bucket (e.g., 1/3 of a second at 30 FPS)
    MIN_PER_BIN = 20

    xs, ys = [], []
    buckets = (bt // BIN) * BIN
    for b in sorted(set(buckets)):
        mm = buckets == b
        if mm.sum() >= MIN_PER_BIN and len(set(yt[mm])) == 2:
            xs.append(b + BIN / 2) # Plot at the center of the bucket
            ys.append(roc_auc_score(yt[mm], p[mm]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, marker="o")
    ax.invert_xaxis()
    ax.axhline(0.5, ls="--", c="grey")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel(f"Steps before end of episode (binned, width {BIN})")
    ax.set_ylabel("Probe AUROC")
    ax.set_title("Lead time to decodable failure (pooled across conditions and seeds, descriptive)")
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "lead_time.png"), dpi=150)

    with open(os.path.join(args.outdir, "probe_auroc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "pooled_auroc", "ci_low", "ci_high", "within_cell_auroc", "gap", "n_valid_splits"])
        for r in results:
            w.writerow(r)
    print(f"\nSaved probe_auroc.csv and lead_time.png to {args.outdir}/")

if __name__ == "__main__":
    main()
