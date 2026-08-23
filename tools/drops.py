#!/usr/bin/env python3
"""
tools/drops.py

This code addresses the question where and when does a policy release the cube?

A release is any sustained gripper opening after the approach-open. In a successful
episode a release inside the cup's angular half-width is the delivery and every other
release is a drop; in a failed episode nothing reached the cup, so every release is a
drop. Where several drops are detected in one episode the first is taken as the drop,
since anything later is a failed re-grasp or the return to home.

Release thresholds (12 deg sustained 5 frames) were calibrated against the recovery
demonstrations, where 20 episodes were performed with a deliberate drop and 30 without.
That calibration is re-run by demo_check() on every invocation.

Gripper values are commanded, not achieved, so a "release" is the policy deciding to
open, which is the quantity of interest.

RUN: python tools/drops.py
"""

import glob
import os

import numpy as np
import pandas as pd

from calibrate_pose import azimuth_fit, BASE_X
from grasp import GRIPPER, OPEN_THR, _merge, _runs
from rollout_paths import discover, parse_policy, CACHE

OUTDIR = "analysis/out_drops"
MIN_RUN = 5                      # frames, suppresses threshold chatter
REL_THR = 12.0                   # release detection, calibrated on recovery training demonstrations
REL_MIN = 5.0                    # 20/20 recall and 0/30 false positives
MIN_TRAVEL = 10.0                # degrees of azimuth travelled since the previous closure. Suppresses
                                 # failed-grasp chatter, where the gripper opens and closes without the
                                 # arm going anywhere. Rollouts only as demonstrations have no failed grasps.

CUP_HALF_WIDTH = np.degrees(np.arctan(1.75 / np.hypot(5.0 - BASE_X, 12.5)))
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv"]

DEMO = "cube-pickup-recovery_20260809_141725"
# 0-indexed episodes of the 20 deliberate-drop demonstrations: demos 2 and 4 of each
# consecutive group of 5 (PROTOCOL.md §3.3), which is 1-indexed 2, 4, 7, 9, ...
EXPECTED = {1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28, 31, 33, 36, 38, 41, 43, 46, 48}

CUP_AZ = np.degrees(np.arctan2(5.0 - BASE_X, 12.5))
T6_AZ = np.degrees(np.arctan2(15.5 - BASE_X, 10.0))
MID_XY = ((15.5 + 5.0) / 2, (10.0 + 12.5) / 2)
MID_AZ = np.degrees(np.arctan2(MID_XY[0] - BASE_X, MID_XY[1]))

def episode_actions(root):
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["episode_index", "frame_index"], kind="stable")

def releases(grip):
    # (approach_end, [(start, end), ...]) for sustained openings after the approach-open
    approach = [r for r in _merge(_runs(grip > OPEN_THR)) if r[1] - r[0] >= MIN_RUN]
    if not approach:
        return None, []
    a_end = approach[0][1]
    rel = [(s + a_end, e + a_end) for s, e in _merge(_runs(grip[a_end:] > REL_THR)) if e - s >= REL_MIN]
    return a_end, rel

def demo_check(to_az):
    # Detector calibration on the recovery demonstrations (PROTOCOL.md §8.22).
    # A clean demo releases once after the approach, into the cup. A drop demo releases
    # twice: the drop, then the delivery. So the detector fires on two or more sustained
    # releases. MIN_TRAVEL is deliberately not applied, since demonstrations contain no
    # failed grasps for it to suppress.
    root = os.path.join(CACHE, DEMO)
    if not os.path.isdir(root):
        print(f"skipping detector calibration, {DEMO} is not in the local cache\n")
        return

    df = episode_actions(root)
    detected, drops = set(), []
    for ep, g in df.groupby("episode_index"):
        A = np.stack(g["action"].to_numpy()).astype(float)
        az = to_az(A[:, 0])
        a_end, rel = releases(A[:, GRIPPER])
        if a_end is None:
            continue
        if len(rel) >= 2:
            detected.add(int(ep))
            s = rel[0][0]
            drops.append({"episode": int(ep), "frame": int(s), "az": float(az[s]), "travel": abs(float(az[s]) - float(az[a_end - 1]))})

    total = df.episode_index.nunique()
    clean = total - len(EXPECTED)
    tp, fp = len(detected & EXPECTED), len(detected - EXPECTED)

    print("--- detector calibration on the recovery demonstration ---")
    print(f"{DEMO}: {total} episodes, {len(EXPECTED)} with a deliberate drop")
    print(f"recall {tp}/{len(EXPECTED)}, false positives {fp}/{clean}")

    dd = pd.DataFrame(drops)
    dd.round(2).to_csv(os.path.join(OUTDIR, "demo_drops.csv"), index=False)
    print(f"demonstrated drop location: n={len(dd)}, median {dd.az.median():.1f} deg, "
          f"IQR {dd.az.quantile(0.25):.1f} to {dd.az.quantile(0.75):.1f}")
    print(f" {int((dd.travel >= MIN_TRAVEL).sum())}/{len(dd)} clear MIN_TRAVEL "
          f"(minimum travel {dd.travel.min():.1f} deg), so the rollout-side filter is a "
          f"no-op on the demonstrations")

    if EXPECTED - detected:
        print(f" missed: {sorted(EXPECTED - detected)}")
    if detected - EXPECTED:
        print(f" false positives: {sorted(detected - EXPECTED)}")
    print()

    
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fit, _, _ = azimuth_fit()
    to_az = lambda p: fit.predict(np.asarray(p, float).reshape(-1, 1))

    demo_check(to_az)

    lab = pd.concat([pd.read_csv(p) for p in TRACKERS], ignore_index=True)
    rows = []
    skips = {"no_label": 0, "no_approach": 0, "no_release": 0, "below_travel": 0}

    for (pol, cell), root in discover().items():
        cond, seed = parse_policy(pol)
        df = episode_actions(root)
        df = df[df.episode_index != 0]

        for ep, g in df.groupby("episode_index"):
            m = lab[(lab.condition == cond) & (lab.seed == seed) & (lab.eval_cell == cell) & (lab.episode == ep)]
            if m.empty:
                skips["no_label"] += 1
                continue

            A = np.stack(g["action"].to_numpy()).astype(float)
            az = to_az(A[:, 0])
            a_end, rel = releases(A[:, GRIPPER])
            if a_end is None:
                skips["no_approach"] += 1
                continue

            if not rel:
                skips["no_release"] += 1
                continue

            bounds = [a_end - 1] + [e - 1 for _, e in rel[:-1]]
            success = int(m.success.iloc[0])
            events = [{"frame": int(s), "az": float(az[s])} for k, (s, _e) in enumerate(rel) if abs(az[s] - az[bounds[k]]) >= MIN_TRAVEL]

            if not events:
                skips["below_travel"] += 1
                continue

            for e in events:
                at_cup = abs(e["az"] - CUP_AZ) <= CUP_HALF_WIDTH
                rows.append({
                    "policy": pol,
                    "condition": cond,
                    "seed": seed,
                    "cell": cell,
                    "episode": int(ep),
                    "success": success,
                    "failure_mode": m.failure_mode.iloc[0],
                    "regrasp": m.regrasp.iloc[0] if "regrasp" in m else None,
                    "kind": "delivery" if (success == 1 and at_cup) else "drop",
                    "frame": e["frame"],
                    "az": round(e["az"], 2),
                    "n_events": len(events)
                })

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit(f"no release events detected. skips: {skips}")
    
    d.to_csv(os.path.join(OUTDIR, "release_events.csv"), index=False)

    n_ep = d.groupby(["policy", "cell", "episode"]).ngroups
    print(f"{len(d)} release events over {n_ep} episodes\n")
    print(f"episodes skipped: {skips}\n")

    print("--- drops per episode, by condition and seed ---")
    per_ep = (d[d.kind == "drop"].groupby(["condition", "seed", "cell", "episode"]).size().rename("drops").reset_index())
    print(per_ep.groupby(["condition", "seed", "cell"])["drops"].agg(episodes_with_drops="size", total_drops="sum").to_string())

    print("\n--- label agreement: episodes with at least 1 detected drop, by scored failure mode ---")
    ep_lvl = (d.assign(has_drop=d.kind.eq("drop")).groupby(["condition", "seed", "cell", "episode", "failure_mode"])["has_drop"].any().reset_index())
    print(pd.crosstab(ep_lvl.failure_mode, ep_lvl.has_drop).to_string())

    DROP_LABELS = {"deliberate_drop", "grasp_drop", "success_after_drop"}
    gated = d[d.failure_mode.isin(DROP_LABELS) & (d.kind == "drop")]
    loc = (gated.sort_values("frame").groupby(["condition", "seed", "cell", "episode"], as_index=False).first())
    loc.to_csv(os.path.join(OUTDIR, "drop_locations.csv"), index=False)

    r = loc[loc.condition == "recovery"]
    print("\n--- recovery drop location, azimuth ---")
    print(f"cup {CUP_AZ:.1f} deg, cube T6 {T6_AZ:.1f} deg, "
          f"midpoint of the carry {MID_AZ:.1f} deg\n")
    print(r.groupby(["seed", "cell"])["az"].agg(n="size", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75)).round(1).to_string())
    print(f"\nall recovery drops: n={len(r)}, median {r.az.median():.1f} deg, "
          f"IQR {r.az.quantile(0.25):.1f} to {r.az.quantile(0.75):.1f}. "
          f"median distance from the carry midpoint {abs(r.az - MID_AZ).median():.1f} deg")

    print(f"\nsaved to {OUTDIR}/")

if __name__ == "__main__":
    main()