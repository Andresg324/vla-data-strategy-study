#!/usr/bin/env python3
"""
tools/rollout_motion.py

Motion statistics for rollout datasets: initiation latency, execution velocity, episode
duration, and an objective check on the no_departure label. Departure is defined as any
arm joint (gripper excluded) deviating more than --min-dev degrees from its first frame.

Writes to analysis/out_motion/:
    rollout_motion_episodes.csv        one row per scored episode
    rollout_motion_summary.csv         by condition, seed and cell
    rollout_motion_by_condition.csv    pooled over cells, feeds Table 2
    no_departure_disagreements.csv     label against telemetry; empty is the pass condition

RUN: python tools/rollout_motion.py
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout_paths import CACHE, discover, parse_policy

TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv"]
OUTDIR = "analysis/out_motion"

def read_actions(root):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "action" not in df.columns:
        return None
    if "timestamp" not in df.columns:
        df["timestamp"] = df["frame_index"] / 30.0
    return df.sort_values(["episode_index", "frame_index"], kind="stable")

def episode_stats(df, min_dev):
    rows = []
    for ep, g in df.groupby("episode_index"):
        A = np.asarray([np.asarray(a, dtype=float) for a in g["action"].to_numpy()])
        t = g["timestamp"].to_numpy(dtype=float)
        if A.ndim != 2 or len(A) < 2:
            continue
        dev = np.abs(A[:, :5] - A[0, :5]).max(axis=1)
        max_dev = float(dev.max())
        idx = np.where(dev > min_dev)[0]
        row = {
            "episode": int(ep),
            "n_frames": len(A),
            "duration_s": float(t[-1] - t[0]),
            "max_dev_deg": round(max_dev, 2),
            "departed": bool(len(idx)),
            "latency_s": np.nan,
            "deg_per_step": np.nan,
            "peak_deg_step": np.nan,
            "exec_s": np.nan,
        }
        if len(idx):
            i0 = idx[0]
            steps = np.abs(np.diff(A[i0:, :5], axis=0))
            row.update({
                "latency_s": round(float(t[i0] - t[0]), 2),
                "deg_per_step": round(float(steps.mean()), 4),
                "peak_deg_step": round(float(steps.max()), 3),
                "exec_s": round(float(t[-1] - t[i0]), 2)
            })
        rows.append(row)
    return pd.DataFrame(rows)


def load_labels():
    frames = []
    for p in TRACKERS:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f" note: {p} not found, labels will be missing for its rows")
    if not frames:
        return pd.DataFrame(columns=["condition", "eval_cell", "seed", "episode"])
    lab = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("condition", "eval_cell", "seed", "episode", "success",
                        "failure_mode", "regrasp", "instance") if c in lab.columns]
    return lab[keep].drop_duplicates(subset=["condition", "eval_cell", "seed", "episode"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="*", help="policy slugs, e.g. clean clean-seed2000")
    ap.add_argument("--cell", nargs="*", help="eval_cells")
    ap.add_argument("--seed", nargs="*", type=int)
    ap.add_argument("--min-dev", type=float, default=20.0, help="degrees of arm-joint deviation, gripper excluded (default 20, chosen by sweep)")
    ap.add_argument("--per-episode", action="store_true", help="print every episode")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    labels = load_labels()
    runs = discover()
    if not runs:
        raise SystemExit(f"no rollout datasets found under {CACHE}")

    out = []
    for (policy, cell), root in runs.items():
        cond, seed = parse_policy(policy)
        if args.policy and policy not in args.policy:
            continue
        if args.cell and cell not in args.cell:
            continue
        if args.seed and seed not in args.seed:
            continue

        df = read_actions(root)
        if df is None:
            print(f" Skipping {os.path.basename(root)}: no action data")
            continue

        st = episode_stats(df, args.min_dev)
        st = st[st.episode != 0]

        if st.empty:
            continue

        lab = labels[(labels.condition == cond) & (labels.seed == seed) & (labels.eval_cell == cell)]
        st = st.merge(lab.drop(columns=["condition", "eval_cell", "seed"], errors="ignore"), on="episode", how="left")
        st.insert(0, "cell", cell)
        st.insert(0, "seed", seed)
        st.insert(0, "condition", cond)
        st.insert(0, "policy", policy)
        out.append(st)

        if args.per_episode:
            print(f"\n--- {os.path.basename(root)} ---")
            print(st.to_string(index=False))

    if not out:
        raise SystemExit("No runs matched those filters")

    ep = pd.concat(out, ignore_index=True)
    ep.to_csv(os.path.join(OUTDIR, "rollout_motion_episodes.csv"), index=False)

    def agg(g):
        s = g[g.success == 1]
        d = g[g.departed]
        return pd.Series({
            "n": len(g),
            "never_departed": int((~g.departed).sum()),
            "latency_median_s": d.latency_s.median(),
            "latency_p90_s": d.latency_s.quantile(0.9),
            "deg_step_all": d.deg_per_step.mean(),
            "deg_step_success": s.deg_per_step.mean(),
            "duration_success_s": s.duration_s.mean(),
            "n_success": len(s),
        })

    summary = ep.groupby(["condition", "seed", "cell"]).apply(agg, include_groups=False).round(3).reset_index()
    summary.to_csv(os.path.join(OUTDIR, "rollout_motion_summary.csv"), index=False)

    by_cond = ep.groupby(["condition", "seed"]).apply(agg, include_groups=False).round(3).reset_index()
    by_cond.to_csv(os.path.join(OUTDIR, "rollout_motion_by_condition.csv"), index=False)

    print("\n--- Per Condition, Seed, and Cell ---")
    print(summary.to_string(index=False))
    print("\n--- Pooled over cells ---")
    print(by_cond.to_string(index=False))

    if "failure_mode" in ep.columns:
        nd = ep[ep.failure_mode == "no_departure"]
        other = ep[(ep.failure_mode.notna()) & (ep.failure_mode != "no_departure")]
        print(f"\n--- No Departure Label Check (threshold {args.min_dev} deg) ---")
        print(f" labeled no departure:          {len(nd)}")
        print(f" ...but measured as departing:  {int(nd.departed.sum())}")
        print(f" labeled something else:        {len(other)}")
        print(f" ...but measured as stationary: {int((~other.departed).sum())}")
        bad = pd.concat([nd[nd.departed], other[~other.departed]])
        bad.to_csv(os.path.join(OUTDIR, "no_departure_disagreements.csv"), index=False)
        if len(bad):
            print("\ndisagreements:")
            print(bad[["policy", "cell", "episode", "failure_mode", "max_dev_deg", "latency_s"]].to_string(index=False))

    print(f"\nSaved to {OUTDIR}")

if __name__ == "__main__":
    main()