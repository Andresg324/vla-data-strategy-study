#!/usr/bin/env python3
"""
tools/show_episode.py
Play one episode out of a chunked LeRobot video.

LeRobot v3 packs many episodes into one mp4 per camera, so an episode is a
time range rather than a file. The episode metadata carries from/to timestamps.

Usage:
    python tools/show_episode.py <dataset_name> <episode_index> [camera_key]
"""

import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/your-hf-username"))

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

if not shutil.which("ffplay"):
    raise SystemExit("ffplay not found. Install with: conda install -c conda-forge ffmpeg")
name = sys.argv[1]
ep = int(sys.argv[2])
cam = sys.argv[3] if len(sys.argv) > 3 else None

root = name if os.path.isdir(name) else os.path.join(CACHE, name)
files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))

if not files:
    files = sorted(
        p for p in glob.glob(os.path.join(root, "meta", "**", "*.parquet"), recursive=True) if "episode" in p.lower()
    )

if not files:
    meta = os.path.join(root, "meta")
    raise SystemExit(
        f"No episode parquet found under {meta}\n"
        f"exists: {os.path.isdir(meta)}\n"
        f"contents: {os.listdir(meta) if os.path.isdir(meta) else 'n/a'}"
    )

df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

m = df[df.episode_index == ep]
if m.empty:
    raise SystemExit(f"episode {ep} not in {name}; available {sorted(df.episode_index.tolist())}")
row = m.iloc[0]


# Demo datasets use overhead/wrist
# Rollouts use camera1/camera2 after the rename.
cams = sorted({c.split("/")[1] for c in df.columns if c.startswith("videos/")})
if cam is None:
    cam = next((c for c in cams if "overhead" in c or "camera1" in c), cams[0])

print("available:", cams, "| using", cam)

chunk = int(row[f"videos/{cam}/chunk_index"])
fidx = int(row[f"videos/{cam}/file_index"])
t0 = float(row[f"videos/{cam}/from_timestamp"])
t1 = float(row[f"videos/{cam}/to_timestamp"])

path = os.path.join(root, "videos", cam, f"chunk-{chunk:03d}", f"file-{fidx:03d}.mp4")
print(f"episode {ep}: {t0:.1f}s to {t1:.1f}s ({t1-t0:.1f}s) in {os.path.basename(path)}")
subprocess.run(["ffplay", "-autoexit", "-ss", str(t0), "-t", str(t1-t0), path])

