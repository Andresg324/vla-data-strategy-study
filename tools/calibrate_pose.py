#!/usr/bin/env python3
"""
Fits joint poses to board (x, y) coordinates from the randomized training demonstrations,
which contain 50 teleoperated grasps at ten known cube positions. Then convert the
rollout grasp poses in analysis/out_endpoints/*.csv into inches.

--apply adds grasp_x and grasp_y columns to analysis/out_endpoints/*.csv in place. 
Re-running endpoints.py drops them, so re-run --apply after it.

usage:
    python tools/calibrate_pose.py                 # fit and report accuracy
    python tools/calibrate_pose.py --apply         # also convert the endpoint CSVs
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from grasp import grasp_pose

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/your-hf-username"))
TRAIN = "cube-pickup-randomized_20260809_115825"    # Training recorded data

JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]

# 10 training positions for the cube, in (x, y) coordinates on the board
T = {1: (2.0, 2.5), 2: (6.5, 7.5), 3: (8.5, 15.0), 4: (12.0, 14.0), 5: (15.5, 2.5),
     6: (15.5, 10.0), 7: (15.5, 14.25), 8: (20.5, 2.5), 9: (20.5, 6.5), 10: (20.5, 10.0)}

BASE_X = 11.0

def azimuth(x, y):
    """Bearing in degrees from the arm base to a board point. 0 is straight ahead."""
    return np.degrees(np.arctan2(np.asarray(x, float) - BASE_X, np.asarray(y, float)))

def load_training():
    root = os.path.join(CACHE, TRAIN)
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable")

    X, Y, eps, skipped = [], [], [], []
    for ep, g in df.groupby("episode_index"):
        A = np.asarray([np.asarray(a, float) for a in g["action"].to_numpy()])
        pose, _, _ = grasp_pose(A)
        if pose is None:
            skipped.append(int(ep))
            continue
        X.append(pose)
        Y.append(T[(int(ep) % 10) + 1])     # To cycle through T1, T2, etc (T1 is episode 0, T10 is episode 9)
        eps.append(int(ep))

    total = df.episode_index.nunique()
    if skipped:
        print(f" {len(skipped)} of {total} episodes had no detectable grasp: {skipped}")
    else:
        print(f" grasp detected in all {total} episodes")
    return np.array(X), np.array(Y), eps

def azimuth_fit():
    """Single source of truth for the pan-to-bearing calibration (PROTOCOL.md §8.23).

    Returns (fit, X, az): the fitted LinearRegression, the (n, 5) grasp poses, and the
    true bearings. Re-reads the training parquet, so call it once per script.
    """
    X, Y, _ = load_training()
    az = azimuth(Y[:, 0], Y[:, 1])
    fit = LinearRegression().fit(X[:, [0]], az)
    return fit, X, az

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--degree", type=int, default=1)
    args = ap.parse_args()

    X, Y, eps = load_training()
    if not np.isfinite(X).all():
        raise SystemExit("non-finite joint values in the training grasps")
    print("shape:", X.shape)
    print(pd.DataFrame(X, columns=JOINTS).describe().round(2).to_string())
    g = pd.DataFrame(X, columns=JOINTS)
    g["pos"] = [f"{a}, {b}" for a, b in Y]

    # This table below is the check on PROTOCOL.md §8.19: the Randomized session crashed and
    # resumed, so the index-to-position mapping below is assumed, not logged. The ten
    # positions span ~100 deg of bearing, so a one-episode offset would blow up the pan
    # column. Spreads of 0.4 to 1.9 deg confirm the mapping held.
    print("\nwithin-position spread (std of the 5 demos at each T):")
    print(g.groupby("pos")[JOINTS].std().round(2).to_string())

    print(f"\n{len(X)} training grasps at {len(set(map(tuple, Y)))} distinct positions")

    model = make_pipeline(
        PolynomialFeatures(args.degree, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 25), cv=5),
    )
    pred = cross_val_predict(model, X, Y, cv=LeaveOneOut())
    err = np.linalg.norm(pred - Y, axis=1)
    print(f"leave-one-out error (degree {args.degree}): median {np.median(err):.2f} in, "
          f"mean {err.mean():.2f} in, 90th percentile {np.percentile(err, 90):.2f} in")
    print("(this is the xy model used by --apply; the bearing calibration the paper "
          "reports is azimuth_fit(), whose LOO is printed by azimuth_analysis.py)")
    print("per-axis MAE: x %.2f in, y %.2f in"
          % tuple(np.abs(pred - Y).mean(axis=0)))

    if not args.apply:
        return

    model.fit(X, Y)
    
    for path in sorted(glob.glob("analysis/out_endpoints/endpoints_*.csv")):
        df = pd.read_csv(path)
        xy = model.predict(df[JOINTS].to_numpy())
        df["grasp_x"], df["grasp_y"] = xy[:, 0].round(2), xy[:, 1].round(2)
        df.to_csv(path, index=False)
        print(f"\n{os.path.basename(path)}")
        print(df.groupby(["cell", "instance"])[["grasp_x", "grasp_y"]].median().round(2).to_string())
        

if __name__ == "__main__":
    main()
