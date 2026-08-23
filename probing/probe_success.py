#!/usr/bin/env python3
"""
probing/probe_success.py

Asks whether the outcome of a rollout is already present in the action expert's
representation before the arm commits, using only the first EARLY_CHUNKS forward
passes of each episode. Yes would mean failure is determined at onset; no would
mean it emerges during execution.

One vector per episode, never per frame, so episode length cannot leak: successes
finish in 13 to 23 s and failures run the full window, and a per-frame probe would
read that off duration alone.

Two controls, both required for the result to mean anything:

  Late window   the same probe on the LAST LATE_CHUNKS passes. Outcome is close to
                explicit there, since the cube is in the cup or it is not, so this
                must score high. A chance result here means the pipeline is broken
                and the early number carries no information either way.

  Within-cell   labels shuffled inside each evaluation cell, PERMS times, rerunning
  permutation   the whole cross-validated probe each time. Cells differ in base rate
                and in appearance, so a probe could score above chance by recognising
                the cell alone. Shuffling within cell destroys the outcome signal and
                preserves the cell structure, so this is the honest null. Cells with
                no outcome variation are dropped outright, since they contribute only
                cell identity and the permutation cannot touch them.

Labels and episode ids come straight out of the .npz written by
extract_activations.py, so there is no join against the trackers to get wrong.

Activations come from model.vlm_with_expert.lm_expert.norm, the action expert's final
RMSNorm, 720 wide, and are one draw from a stochastic policy. Two extractions of the
same episode correlate at about 0.998, so these numbers carry extraction noise on top
of probe noise. extract_activations.py seeds per episode so a rerun reproduces.

RUN: python probing/probe_success.py
"""

import glob
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INDIR = "probing/out_np"
OUTDIR = "analysis/out_probe"

EARLY_CHUNKS = 3        # first three forward passes, frames 0 to 100, before the grasp
LATE_CHUNKS = 3         # final passes
N_COMP = 10             # PCA components; n is small on purpose as there are 40-90 episodes
PERMS = 500             # permutations per point estimate; resolves p to about 0.002
MIN_MINORITY = 6        # AUC and Stratified folds are meaningless if there are 5 or less of the rarer outcome with an N of 30

WINDOW = 3              # passes per sweep window, same width as EARLY_CHUNKS
SWEEP_STEP = 1          # offsets increment, in passes
CHUNK_S = 50 / 30       # One action chunk (50 frames at 30 FPS)
GRASP_S = 215 / 30      # Gripper tends to close at around 215 frames in distribution

def window_features(X, uid, tfe, n_chunks, which):
    """One vector per episode: the mean hidden state over the first or last
    n_chunks forward passes. t_from_end counts down, so the largest values are
    the earliest passes in the episode."""
    feats, ids = [], []
    for u in sorted(set(uid.tolist())):
        m = uid == u
        t = tfe[m]
        order = np.argsort(-t) if which == "early" else np.argsort(t)
        feats.append(X[m][order[:n_chunks]].mean(axis=0))
        ids.append(u)
    return np.vstack(feats), np.array(ids)

def auc_cv(F, yb, seed=0):
    """Cross-validated AUC. Returns None when a stratified fold cannot be built."""
    counts = np.bincount(yb, minlength=2)
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return None
    n_comp = int(min(N_COMP, F.shape[0] - 1, F.shape[1]))
    model = make_pipeline(StandardScaler(), PCA(n_comp, svd_solver="full"), LogisticRegression(C=0.1, max_iter=5000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        p = cross_val_predict(model, F, yb, cv=cv, method="predict_proba")[:, 1]
    assert np.isfinite(p).all(), "probe produced non-finite probabilities"
    return float(roc_auc_score(yb, p))

def perm_null(F, yb, cells, n_perm=PERMS, seed=0):
    """AUCs under labels shuffled within each evaluation cell."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_perm):
        yp = yb.copy()
        for c in np.unique(cells):
            m = cells == c
            yp[m] = rng.permutation(yb[m])
        a = auc_cv(F, yp, seed=i)
        if a is not None:
            out.append(a)
    return np.array(out)

def window_at(X, uid, tfe, offset, width):
    """One vector per episode: the mean hidden state over passes
    [offset, offset + width), counting from the start of the episode. Episodes with
    fewer than offset + width passes are omitted, so n falls as the offset grows and
    the surviving episodes skew toward failures, which run the full window. The
    within-cell permutation null is recomputed on each offset's own subset, so it
    absorbs that shift in composition."""
    feats, ids = [], []
    for u in sorted(set(uid.tolist())):
        m = uid == u
        idx = np.argsort(-tfe[m])           # Earliest pass first
        if len(idx) < offset + width:
            continue
        feats.append(X[m][idx[offset:offset + width]].mean(axis=0))
        ids.append(u)
    if not feats:
        return None, np.array([])
    return np.vstack(feats), np.array(ids)

def sweep_policy(slug, X, uid, tfe, y_all, cell, n_perm):
    """AUC against offset, from the start of the episode until class balance dies."""
    info = {}
    for i, u in enumerate(uid):
        info.setdefault(int(u), (int(y_all[i]), str(cell[i])))
    longest = max(int((uid == u).sum()) for u in set(uid.tolist()))

    rows = []
    n0 = None
    for off in range(0, longest - WINDOW + 1, SWEEP_STEP):
        F, ids = window_at(X, uid, tfe, off, WINDOW)
        if F is None:
            break
        yb = np.array([info[int(u)][0] for u in ids])
        ce = np.array([info[int(u)][1] for u in ids])
        counts = np.bincount(yb, minlength=2)
        if counts.min() < MIN_MINORITY:
            break
        a = auc_cv(F, yb)
        if a is None:
            break
        null = perm_null(F, yb, ce, n_perm=n_perm)
        if n0 is None:
            n0 = len(yb)
        rows.append({
            "policy": slug,
            "offset": off,
            "t_start_s": round(off * CHUNK_S, 2),
            "t_end_s": round((off + WINDOW) * CHUNK_S, 2),
            "episodes": len(yb),
            "complete": int(len(yb) == n0),
            "successes": int(counts[1]),
            "failures": int(counts[0]),
            "auc": round(a, 3),
            "null_median": round(float(np.median(null)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "perm_p": round((1 + int((null >= a).sum())) / (1 + len(null)), 4),
            "n_perms": len(null),
        })

        print(f" offset {off:2d} ({off * CHUNK_S:5.1f}s) n={len(yb):3d} "
              f"({counts[1]}s/{counts[0]}f) AUC {a:.3f} null p95 {rows[-1]['null_p95']:.3f}")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also sweep the window across the episode, writing success_sweep.csv")
    ap.add_argument("--perms", type=int, default=PERMS)
    ap.add_argument("--sweep-perms", type=int, default=200, help="fewer permutations per sweep point as there are many of them")
    args = ap.parse_args()


    os.makedirs(OUTDIR, exist_ok=True)
    rows, sweep_rows = [], []

    for path in sorted(glob.glob(os.path.join(INDIR, "activations_*.npz"))):
        slug = os.path.basename(path)[len("activations_"):-len(".npz")]
        d = np.load(path, allow_pickle=True)
        for k in ("X", "episode", "t_from_end", "eval_cell", "success"):
            if k not in d:
                raise SystemExit(f"{path} has no {k}; re-extract with extract_activations.py")

        X = d["X"].astype(np.float64)
        uid = d["episode"].astype(int)          # episode ID already unique across cells
        tfe = d["t_from_end"].astype(float)
        cell = d["eval_cell"].astype(str)
        y_all = d["success"].astype(int)

        # Drop any cell with no outcome variation: it would contribute cell identity
        # and nothing else, and the within-cell permutation cannot touch it.
        keep = np.ones(len(y_all), bool)
        for c in np.unique(cell):
            sel = cell == c
            if len(np.unique(y_all[sel])) < 2:
                keep &= ~sel

        if "layer" in d:
            print(f"  {slug}: layer {str(d['layer'][0])}")

        if not keep.any():
            print(f"{slug}: no cell has outcome variation, skipped")
            continue

        X, uid, tfe, cell, y_all = X[keep], uid[keep], tfe[keep], cell[keep], y_all[keep]
        keepdim = X.std(axis=0) > 0
        X = X[:, keepdim]

        # Collapse to one row per episode
        F_early, ids = window_features(X, uid, tfe, EARLY_CHUNKS, "early")
        F_late, _ = window_features(X, uid, tfe, LATE_CHUNKS, "late")

        info = {}
        for i, u in enumerate(uid):
            info.setdefault(int(u), (int(y_all[i]), str(cell[i])))
        yb = np.array([info[int(u)][0] for u in ids])
        cells_ep = np.array([info[int(u)][1] for u in ids])

        counts = np.bincount(yb, minlength=2)
        if counts.min() < MIN_MINORITY:
            print(f"{slug}: only {counts.min()} episodes in the minority outcome, skipped")
            continue

        auc_early = auc_cv(F_early, yb)
        auc_late = auc_cv(F_late, yb)
        null = perm_null(F_early, yb, cells_ep, n_perm=args.perms)
        p = (1 + int((null >= auc_early).sum())) / (1 + len(null))
        
        rows.append({
            "policy": slug,
            "episodes": len(yb),
            "cells": "+".join(sorted(set(cells_ep))),
            "successes": int(counts[1]),
            "failures": int(counts[0]),
            "auc_early": round(auc_early, 3),
            "auc_late_control": round(auc_late, 3) if auc_late is not None else None,
            "null_median": round(float(np.median(null)), 3),
            "null_p95": round(float(np.percentile(null, 95)), 3),
            "perm_p": round(p, 4),
            "n_perms": len(null),
        })
        print(f"{slug}: {len(yb)} episodes ({counts[1]}s/{counts[0]}f), "
              f"early AUC {auc_early:.3f}, late AUC {auc_late:.3f}, "
              f"null median {np.median(null):.3f}, p = {p:.4f}")

        if args.sweep:
            print(f" sweeping {slug}")
            sweep_rows.extend(sweep_policy(slug, X, uid, tfe, y_all, cell, args.sweep_perms))

    if not rows:
        raise SystemExit("no policy had enough outcome variation to probe.")

    t = pd.DataFrame(rows)
    print("\n" + t.to_string(index=False))
    t.to_csv(os.path.join(OUTDIR, "success_probe.csv"), index=False)

    if sweep_rows:
        s = pd.DataFrame(sweep_rows)
        s.to_csv(os.path.join(OUTDIR, "success_sweep.csv"), index=False)
        print(f"\nsaved to {OUTDIR}/success_sweep.csv")

    print(f"\nsaved to {OUTDIR}/success_probe.csv")


if __name__ == "__main__":
    main()
