#!/usr/bin/env python3
"""
Regresses the true cube bearing on the action expert's final hidden state,
using the New Positions rollouts, where the cube occupies five locations
within a single session so there is no session confound.

Clean and Randomized are each other's control: same architecture, same task,
same scene, same episodes, different training data. Behaviorally Clean's aim does not
move with the cube (spread 0.2 and 1.6 deg across cells) and Randomized's does (10.2
and 24.3 deg). The question is whether the representations agree.

Positive control: the same probe predicting t_from_end, which must be decodable
if the extraction is sound. A null on position with a null on the control means
the pipeline is broken, not that the information is absent.

Activations come from model.vlm_with_expert.lm_expert.norm, the action expert's final
RMSNorm, 720 wide, and are one draw from a stochastic policy. Two extractions of the
same episode correlate at about 0.998, so these numbers carry extraction noise on top
of probe noise. extract_activations.py seeds per episode so a rerun reproduces.

RUN: python probing/probe_position.py
"""

import glob
import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INDIR = "probing/out_np"
OUTDIR = "analysis/out_probe"
BASE_X = 11.0
POS = {"E1": (2.0, 7.5), "E2": (6.5, 2.5), "E3": (12.0, 10.0), "E4": (15.5, 6.5), "E5": (19.5, 13.5)}
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv"]

az = lambda x, y: np.degrees(np.arctan2(x - BASE_X, y))
AZ = {k: az(*v) for k, v in POS.items()}

def probe(X, y, groups, n_comp=50, n_splits=5):
    # Ridge on PCA components, grouped by episode; returns median | error |
    n_comp = min(n_comp, X.shape[0] - 1, X.shape[1])

    # Outer CV is grouped; RidgeCV's inner alpha selection is not. That leaks across
    # frames within an episode for alpha choice only, never for the reported error.
    model = make_pipeline(StandardScaler(), PCA(n_comp, svd_solver="full"), RidgeCV(alphas=np.logspace(-1, 5, 25), cv=5))

    # sklearn's internals raise FPU flags on this data, inputs are verified finite and predictions are
    # asserted finite below, so the flags are perceived noise
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits))
    assert np.isfinite(pred).all(), "probe produced non-finite predictions"
    return np.abs(pred - y), pred

def cluster_bootstrap_ci(errors, groups, n_boot=2000, seed=0):
    """Percentile CI on the median absolute error, resampling whole groups.

    Samples inside one episode (or one position) are correlated, so the resampling
    unit is the group, not the frame. `errors` and `groups` are 1-D arrays of equal
    length holding out-of-fold absolute errors and the group each sample came from.
    """
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    meds = np.empty(n_boot)

    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        meds[b] = np.median(np.concatenate([errors[idx[g]] for g in pick]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

def main():
    os.makedirs(OUTDIR, exist_ok = True)
    lab = pd.concat([pd.read_csv(p) for p in TRACKERS], ignore_index=True)
    rows = []

    for path in sorted(glob.glob(os.path.join(INDIR, "activations_*.npz"))):
        slug = os.path.basename(path)[len("activations_"):-len(".npz")]
        d = np.load(path, allow_pickle=True)
        if "ep_true" not in d:
            raise SystemExit(f"{path} has no ep_true; re-extract with the patch")

        keep = d["eval_cell"] == "new_positions"
        if keep.sum() == 0:
            print(f"{slug}: no new_positions activations, skipped")
            continue

        X = d["X"][keep].astype(np.float64)
        keepdim = X.std(axis=0) > 0
        X = X[:, keepdim]
        ep = d["ep_true"][keep]
        tfe = d["t_from_end"][keep].astype(float)
        cond, seed = str(d["condition"][keep][0]), int(d["seed"][keep][0])

        m = lab[(lab.condition == cond) & (lab.seed == seed) & (lab.eval_cell == "new_positions")][["episode", "instance"]]
        inst = dict(zip(m.episode, m.instance))
        y = np.array([AZ.get(inst.get(int(e)), np.nan) for e in ep])
        ok = ~np.isnan(y)
        X, ep, tfe, y = X[ok], ep[ok], tfe[ok], y[ok]

        pos = np.array([inst.get(int(e), "?") for e in ep])

        chance = np.abs(y - y.mean())
        n_pos = len(set(pos))
        err_ep, _ = probe(X, y, ep)
        err_pos, _ = probe(X, y, pos, n_splits=min(5, n_pos))
        cerr, _ = probe(X, tfe, ep)

        ep_lo, ep_hi = cluster_bootstrap_ci(err_ep, ep)
        pos_lo, pos_hi = cluster_bootstrap_ci(err_pos, pos)

        rows.append({
            "policy": slug,
            "episodes": len(set(ep)),
            "samples": len(y),
            "n_positions": n_pos,
            "chance_med_deg": round(float(np.median(chance)), 1),
            "probe_by_episode_deg" : round(float(np.median(err_ep)), 1),
            "probe_by_episode_lo"  : round(ep_lo, 1),
            "probe_by_episode_hi"  : round(ep_hi, 1),
            "probe_by_position_deg": round(float(np.median(err_pos)), 1),
            "probe_by_position_lo" : round(pos_lo, 1),
            "probe_by_position_hi" : round(pos_hi, 1),
            "probe_mean_deg": round(float(err_ep.mean()), 1),
            "control_tfe_med_steps": round(float(np.median(cerr)), 1),
            "control_chance_steps": round(float(np.median(np.abs(tfe - tfe.mean()))), 1),
        })
        print(f"{slug}: {len(set(ep))} episodes, {len(y)} samples, "
              f"by-episode {np.median(err_ep):.1f} [{ep_lo:.1f}, {ep_hi:.1f}], "
              f"by-position {np.median(err_pos):.1f} [{pos_lo:.1f}, {pos_hi:.1f}], chance {np.median(chance):.1f} deg")

    t = pd.DataFrame(rows)
    print("\n" + t.to_string(index=False))
    t.to_csv(os.path.join(OUTDIR, "position_probe.csv"), index=False)
    print(f"\nsaved to {OUTDIR}/")

if __name__ == "__main__":
    main()