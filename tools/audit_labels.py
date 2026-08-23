#!/usr/bin/env python3
"""
tools/audit_labels.py

Independent screens over every scored episode. Two jobs: audit the live success labels
(PROTOCOL.md §8.27) and measure the recording window (§4.12). Every screen is derived from
telemetry rather than from the labels being tested.

  Window   Episode lengths give the recording ceiling, from which the chunk count and the
           split between executed and inference time follow.
  Screen 1 Duration. A success is terminated when achieved; a failure runs the window out.
           A success at the ceiling, or a failure well short of it, is anomalous.
  Screen 2 Every success must contain a gripper release inside the cup's angular
           half-width. One that does not is a candidate mis-score.
  Screen 3 The mirror: failures containing a short-duration cup release, which is what a
           success looks like. Catches errors in the other direction.
  Misses   For every drop-labelled episode the detector missed, why it was missed.

RUN: python tools/audit_labels.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from calibrate_pose import azimuth_fit
from grasp import GRIPPER
from rollout_paths import discover, parse_policy
from drops import CUP_AZ, CUP_HALF_WIDTH, MIN_TRAVEL, episode_actions, releases

OUTDIR = "analysis/out_audit"
TRACKERS = ["documents/results_full.csv", "documents/exploratory.csv"]
DROP_LABELS = {"deliberate_drop", "grasp_drop", "success_after_drop"}

FPS = 30            # dataset.fps, PROTOCOL.md §4.12
CHUNK = 50          # n_action_steps, PROTOCOL.md §4.12
WINDOW_S = 45.0     # episode_time_s, wall clock, PROTOCOL.md §4.10

NEAR_CEILING = 2.0  # a success within this many seconds of the ceiling is flagged
EARLY = 3.0         # a failure ending this many seconds before the ceiling is flagged


def collect(to_az):
    """One row per scored episode, with the telemetry the screens need."""
    lab = pd.concat([pd.read_csv(p) for p in TRACKERS], ignore_index=True)
    rows, unlabelled = [], 0

    for (pol, cell), root in discover().items():
        cond, seed = parse_policy(pol)
        df = episode_actions(root)
        df = df[df.episode_index != 0]

        for ep, g in df.groupby("episode_index"):
            m = lab[(lab.condition == cond) & (lab.seed == seed)
                    & (lab.eval_cell == cell) & (lab.episode == ep)]
            if m.empty:
                unlabelled += 1
                continue

            A = np.stack(g["action"].to_numpy()).astype(float)
            az = to_az(A[:, 0])
            if "timestamp" in g:
                t = g["timestamp"].to_numpy(float)
                dur = float(t[-1] - t[0])
            else:
                dur = len(A) / FPS

            a_end, rel = releases(A[:, GRIPPER])
            if a_end is None:
                near, max_travel, n_rel, n_kept = np.nan, np.nan, 0, 0
            else:
                bounds = [a_end - 1] + [e - 1 for _, e in rel[:-1]]
                # Deliberately not travel-filtered, unlike drops.py: the audit wants every
                # opening the detector can see, including ones drops.py would suppress.
                cup_d = [abs(float(az[s]) - CUP_AZ) for s, _ in rel]
                travel = [abs(float(az[s]) - float(az[bounds[k]]))
                          for k, (s, _e) in enumerate(rel)]
                near = min(cup_d) if cup_d else np.nan
                max_travel = max(travel) if travel else np.nan
                n_rel = len(rel)
                n_kept = sum(1 for v in travel if v >= MIN_TRAVEL)

            rows.append({
                "policy": pol,
                "condition": cond,
                "seed": seed,
                "cell": cell,
                "episode": int(ep),
                "success": int(m.success.iloc[0]),
                "failure_mode": str(m.failure_mode.iloc[0]),
                "frames": len(A), 
                "duration_s": round(dur, 3),
                "n_releases": n_rel, 
                "n_kept": n_kept,
                "nearest_release_deg": near,
                "max_travel_deg": max_travel,
            })

    if unlabelled:
        print(f"warning: {unlabelled} recorded episodes had no tracker row\n")
    return pd.DataFrame(rows)


def window(ep):
    print("=== recording window (PROTOCOL.md §4.12) ===")
    ceil_f = int(ep.frames.max())
    ceil_s = ceil_f / FPS
    chunks = ceil_f / CHUNK
    inference = WINDOW_S - ceil_s
    print(f"longest episode      {ceil_f} frames = {ceil_s:.2f} s of recorded motion at {FPS} fps")
    print(f"chunks of {CHUNK}         {chunks:.2f}")
    print(f"episodes at ceiling  {int((ep.frames == ceil_f).sum())} of {len(ep)}")
    if abs(chunks - round(chunks)) < 0.05:
        print(f"inference time       {inference:.2f} s over {chunks:.0f} passes = "
              f"{inference / chunks * 1000:.0f} ms each")
    print(f"action-update rate   {CHUNK / FPS:.3f} s recorded ({FPS / CHUNK:.3f} Hz), "
          f"{WINDOW_S / chunks:.2f} s wall clock ({chunks / WINDOW_S:.3f} Hz)\n")


def screen_duration(ep):
    print("=== screen 1: duration ===")
    ceil_s = ep.duration_s.max()
    print(ep.groupby("success").duration_s.describe().round(1).to_string())

    hi = ep[(ep.success == 1) & (ep.duration_s > ceil_s - NEAR_CEILING)]
    lo = ep[(ep.success == 0) & (ep.duration_s < ceil_s - EARLY)]
    cols = ["policy", "cell", "episode", "failure_mode", "duration_s"]
    print(f"\n{len(hi)} successes within {NEAR_CEILING:.0f} s of the {ceil_s:.1f} s ceiling:")
    print(hi[cols].sort_values("duration_s", ascending=False).to_string(index=False))
    print(f"\n{len(lo)} failures ending more than {EARLY:.0f} s early:")
    print(lo[cols].sort_values("duration_s").to_string(index=False))
    return hi, lo


def screen_cup(ep):
    print(f"\n=== screens 2 and 3: release inside the cup window "
          f"(+/- {CUP_HALF_WIDTH:.1f} deg of {CUP_AZ:.1f}) ===")
    cols = ["policy", "cell", "episode", "failure_mode", "nearest_release_deg", "duration_s"]

    s = ep[ep.success == 1]
    inside = s[s.nearest_release_deg <= CUP_HALF_WIDTH]
    print(f"successes: {len(inside)} of {len(s)} contain a cup release, "
          f"median {inside.nearest_release_deg.median():.1f} deg from the cup centre "
          f"(IQR {inside.nearest_release_deg.quantile(.25):.1f} to "
          f"{inside.nearest_release_deg.quantile(.75):.1f})")
    s_bad = s[~(s.nearest_release_deg <= CUP_HALF_WIDTH)]
    print(f"\n{len(s_bad)} successes with NO cup release:")
    print(s_bad[cols].sort_values("nearest_release_deg", ascending=False).to_string(index=False))

    f = ep[ep.success == 0]
    f_hit = f[f.nearest_release_deg <= CUP_HALF_WIDTH]
    ceil_s = ep.duration_s.max()
    print(f"\n{len(f_hit)} of {len(f)} failures contain a cup release "
          f"({int((f_hit.duration_s > ceil_s - NEAR_CEILING).sum())} ran to the ceiling, "
          f"which §6.7 permits):")
    print(f_hit[cols].sort_values("duration_s").to_string(index=False))
    return s_bad, f_hit


def misses(ep):
    print("\n=== drop-labelled episodes the detector missed, and why ===")
    md = ep[ep.failure_mode.isin(DROP_LABELS) & (ep.n_kept == 0)]
    tot = int(ep.failure_mode.isin(DROP_LABELS).sum())
    print(f"{len(md)} of {tot} drop-labelled episodes have no release surviving "
          f"MIN_TRAVEL = {MIN_TRAVEL}")
    print(md[["policy", "cell", "episode", "failure_mode", "n_releases",
              "max_travel_deg", "duration_s"]]
          .sort_values("max_travel_deg", ascending=False).to_string(index=False))
    if len(md) and md.n_releases.min() > 0:
        print(f"\nall {len(md)} produced a release; none travelled far enough "
              f"(largest {md.max_travel_deg.max():.1f} deg against a {MIN_TRAVEL} deg "
              f"threshold), so the misses are travel-filtered, not undetected openings")
    return md


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    fit, _, _ = azimuth_fit()
    to_az = lambda p: fit.predict(np.asarray(p, float).reshape(-1, 1))

    ep = collect(to_az)
    ep.to_csv(os.path.join(OUTDIR, "episode_audit.csv"), index=False)
    print(f"{len(ep)} scored episodes\n")

    window(ep)
    hi, lo = screen_duration(ep)
    s_bad, f_hit = screen_cup(ep)
    md = misses(ep)

    flagged = pd.concat([
        hi.assign(screen="success at ceiling"),
        lo.assign(screen="failure ended early"),
        s_bad.assign(screen="success without cup release"),
        f_hit[f_hit.duration_s < ep.duration_s.max() - NEAR_CEILING]
            .assign(screen="short failure with cup release"),
    ], ignore_index=True)
    flagged.to_csv(os.path.join(OUTDIR, "flagged_episodes.csv"), index=False)
    md.to_csv(os.path.join(OUTDIR, "detector_misses.csv"), index=False)

    print("\n=== review queue: episodes flagged by more than one screen ===")
    both = (flagged.groupby(["policy", "cell", "episode"]).size()
            .rename("n_screens").reset_index().query("n_screens > 1"))
    print(both.to_string(index=False) if len(both) else "none")
    print(f"\nsaved to {OUTDIR}/")


if __name__ == "__main__":
    main()