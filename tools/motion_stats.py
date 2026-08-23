#!/usr/bin/env python3
"""
tools/motion_stats.py
Checks the recordings for motion speed and corrective sub-movements.

Episode duration alone doesn't separate these two. Mean per-step joint displacement
can: if two conditions move at the same speed but one finishes sooner, the
shorter one simply contains less hesitation and re-adjustment.

deg/step here is the mean absolute per-frame change across all six commanded joints, 
gripper included, since the question is how fast the teleoperator moved. 
rollout_motion.py reports a same-named statistic over the five arm joints only. Do not 
compare the two directly.

Usage:
    python tools/motion_stats.py cube-pickup-clean_20260809_105745 \
                                 cube-pickup-color_20260809_183224
"""

import os
import sys
import json

import numpy as np
import glob
import pandas as pd

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/your-hf-username"))
STEPS, BATCH = 10000, 32
OUTDIR = "analysis/out_pace"

def dataset_fps(root, default=30):
    p = os.path.join(root, "meta", "info.json")
    if os.path.exists(p):
        return float(json.load(open(p))["fps"])
    print(f" note: no meta/info.json under {root}, assuming {default} fps")
    return default

def load_actions(root):
    #Actions per frame and the episode ID of each frame
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}/data")

    df = pd.concat(
        [pd.read_parquet(f, columns=["action", "episode_index", "frame_index"]) for f in files], ignore_index=True,
    )

    df = df.sort_values(["episode_index", "frame_index"], kind="stable")
    # Read straight from the underlying table, not the video

    actions = np.stack(df["action"].to_numpy())
    episodes = df["episode_index"].to_numpy()
    return actions, episodes

def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for name in sys.argv[1:]:
        root = name if os.path.isdir(name) else os.path.join(CACHE, name)
        actions, episodes = load_actions(root)
        fps = dataset_fps(root)

        # Episode 0 is kept. The warm-up convention applies to rollouts only; in a training
        # dataset episode 0 is demonstration T1, which calibrate_pose.py maps by index.
        per_step, durations = [], []
        for ep in np.unique(episodes):
            a = actions[episodes == ep]
            if len(a) < 2:
                continue

            per_step.append(np.abs(np.diff(a, axis=0)).mean())
            durations.append(len(a)/fps)

        frames = int(len(actions))
        rows.append({
            "dataset": os.path.basename(root),
            "episodes": len(durations),
            "fps": fps,
            "frames": frames,
            "frames_per_ep": frames / len(durations),
            "sec_per_ep": float(np.mean(durations)),
            "deg_per_step": float(np.mean(per_step)),
            "deg_per_sec": float(np.mean(per_step)) * fps,
            "epochs_at_10k_steps": STEPS * BATCH / frames,
        })

    t = pd.DataFrame(rows)
    base = t.loc[t.dataset.str.contains("clean_2026"), "sec_per_ep"]
    if len(base):
        t["vs_clean_pct"] = (t.sec_per_ep / float(base.iloc[0]) - 1) * 100
    else:
        print(" note: no 'clean' dataset in this run, vs_clean_pct omitted from pace.csv")
    t.round(4).to_csv(os.path.join(OUTDIR, "pace.csv"), index=False)
    print(t.round(3).to_string(index=False))
    print(f"\nsaved to {OUTDIR}/pace.csv")

if __name__ == "__main__":
    main()