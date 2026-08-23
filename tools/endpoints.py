#!/usr/bin/env python3
"""
For each episode, finds the first close-on-target event (gripper opens, then
closes) and records the five arm joints at that instant. If a policy is doing
open-loop trajectory replay, this pose is the same regardless of where the cube
actually is. Goal is to find whether changing the location of the cube by 1 or 2
inches affects the approach/trajectory of the gripper.

usage:
    python tools/endpoints.py --policy clean --cells in_distribution near_1in near_2in
    python tools/endpoints.py --policy randomized --cells new_positions --per-episode
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from grasp import grasp_pose
from rollout_paths import discover, parse_policy

JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv"]

def actions(root):
    f = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(x) for x in f], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"], kind="stable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--per-episode", action="store_true")
    ap.add_argument("--shift-from", help="cell to use as the comparison baseline; "
                                          "requires every cell to hold the cube at a single fixed position")
    args = ap.parse_args()

    condition, seed = parse_policy(args.policy)
    runs = discover()
    lab = pd.concat([pd.read_csv(p) for p in TRACKERS], ignore_index=True)

    rows, no_grasp, multi, seen = [], [], [], []
    for cell in args.cells:
        root = runs.get((args.policy, cell))
        if root is None:
            print(f"skipping {cell}: no local dataset found")
            continue
        seen.append(cell)
        df = actions(root)
        for ep, g in df.groupby("episode_index"):
            if ep == 0:
                continue   # First episode is a warm-up, per protocol

            A = np.asarray([np.asarray(a, float) for a in g["action"].to_numpy()])
            pose, idx, released = grasp_pose(A)
            if pose is None:
                no_grasp.append((cell, int(ep)))
                continue

            m = lab[
                (lab.condition == condition) & 
                (lab.seed == seed) & 
                (lab.eval_cell == cell) & 
                (lab.episode == ep)
            ]
            if len(m) > 1:
                multi.append((cell, int(ep), len(m)))
            row = {
                "cell": cell,
                "episode": int(ep),
                "frame": idx,
                "success": int(m.success.iloc[0]) if len(m) and pd.notna(m.success.iloc[0]) else -1,
                "instance": m.instance.iloc[0] if len(m) else "-",
                "released": released
            }
            row.update(dict(zip(JOINTS, pose.round(2))))
            rows.append(row)
    if no_grasp:
        print(f"no grasp detected in {len(no_grasp)} episodes: {no_grasp}")

    if multi:
        print(f"WARNING: {len(multi)} episodes matched more than one tracker row "
              f"(results_full.csv and exploratory.csv overlap): {multi}")

    ep_df = pd.DataFrame(rows)
    if ep_df.empty:
        raise SystemExit("No grasp events found" + ("" if seen else f", no local dataset for any of {args.cells}"))

    unmatched = int((ep_df.success == -1).sum())
    if unmatched:
        print(f"WARNING: {unmatched} of {len(ep_df)} episodes found no tracker row "
              f"(condition={condition!r}, seed={seed}). Check the label join.")

    os.makedirs("analysis/out_endpoints", exist_ok=True)
    out = f"analysis/out_endpoints/endpoints_{args.policy}.csv"
    if os.path.exists(out):
        prev = pd.read_csv(out)
        prev = prev[~prev.cell.isin(seen)]        # Replaces only regenerated cells

        # calibrate_pose.py --apply writes these back in; drop them so the merged
        # file is not half-populated. Re-run --apply after this.
        prev = prev.drop(columns=[c for c in ("grasp_x", "grasp_y") if c in prev.columns])
        ep_df = pd.concat([prev, ep_df], ignore_index=True)
        ep_df = ep_df.sort_values(["cell", "episode"], kind="stable")

    ep_df.to_csv(out, index=False)
    print(f"cells now in {os.path.basename(out)}: {sorted(ep_df.cell.unique())}")


    if args.per_episode:
        print(ep_df.to_string(index=False))

    print("\nreleases vs a grasp with no releases:")
    print(ep_df.groupby(["cell", "released"]).size().to_string())
    print(f"\n--- median and IQRs of grasp pose by cell, {args.policy} ---")
    print(ep_df.groupby("cell")[JOINTS].agg(["median", "count"]).round(2).to_string())
    q = ep_df.groupby("cell")[JOINTS].quantile([0.25, 0.75]).round(2)
    print(q.to_string())

    if ep_df.cell.nunique() > 1:
        med = ep_df.groupby("cell")[JOINTS].median()                
        q1 = ep_df.groupby("cell")[JOINTS].quantile(0.25)
        q3 = ep_df.groupby("cell")[JOINTS].quantile(0.75)                
        spread = ((q3 - q1) / 1.349).replace(0, np.nan) # IQRS rescaled to an SD equivalent
        base = args.shift_from
        if base and base in med.index:
            print(f"\n--- shift from {base}: degrees, and in SDs of {base} ---")
            for c in med.index:
                if c == base:
                    continue

                d = med.loc[c] - med.loc[base]
                z = d / spread.loc[base]
                print(f"{c:16s} " + " ".join(f"{j}={d[j]:+6.2f} ({z[j]:+.1f}s)" for j in JOINTS))
        elif base:
            print(f"\n(no rows for --shift-from {base}, skipping shift table)")

    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()
