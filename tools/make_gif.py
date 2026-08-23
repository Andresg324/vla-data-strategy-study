#!/usr/bin/env python3
"""
tools/make_gif.py — export one episode from a LeRobot v3 dataset as an optimised gif.

LeRobot v3 packs many episodes into one mp4 per camera, so an episode is a time range.
This reads the episode's boundaries from the dataset metadata, cuts that range out of the
right chunk file, and runs a two-pass palette encode so the gif stays small enough for a
README.

usage:
    python tools/make_gif.py <dataset_name> <episode_index> [options]

    python tools/make_gif.py rollout_color_in_distribution_20260810_143012 7 \
        --speed 2 --width 600 --out media/overhead_demo.gif

options:
    --cam     camera key or suffix, default camera1 (overhead)
    --speed   playback multiplier, default 2.0
    --fps     output frame rate, default 12
    --width   output width in px, height follows aspect, default 600
    --start   seconds to skip from the start of the episode, default 0
    --dur     seconds to keep, 0 means to the end of the episode
    --out     output path, default media/<dataset>_ep<N>.gif
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

CACHE = os.environ.get(
    "LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/your-hf-username")
)


def episode_meta(root):
    files = sorted(
        glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True)
    )
    if not files:
        sys.exit(f"no episode metadata under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def col_for(df, cam, key):
    hits = [c for c in df.columns if cam in c and key in c]
    return hits[0] if hits else None


def resolve_video(root, df, row, cam):
    """Return (path, t0, t1) for this episode's slice of the chunked video."""
    ck = col_for(df, cam, "chunk_index")
    fk = col_for(df, cam, "file_index")
    if ck is None or fk is None:
        sys.exit(f"no video columns matching '{cam}'. columns:\n  " + "\n  ".join(df.columns))

    # the metadata column name carries the full camera key, e.g.
    # "videos/observation.images.camera1/chunk_index"
    cam_key = ck.split("/")[1] if "/" in ck else f"observation.images.{cam}"
    path = os.path.join(
        root, "videos", cam_key, f"chunk-{int(row[ck]):03d}", f"file-{int(row[fk]):03d}.mp4"
    )
    if not os.path.exists(path):
        found = glob.glob(os.path.join(root, "videos", cam_key, "**", "*.mp4"), recursive=True)
        sys.exit(f"expected {path}\nfound instead:\n  " + "\n  ".join(found))

    f0 = col_for(df, cam, "from_timestamp")
    f1 = col_for(df, cam, "to_timestamp")
    if f0 and f1:
        return path, float(row[f0]), float(row[f1])

    # fallback: derive the offset from cumulative episode lengths within this file
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    fps = float(info["fps"])
    same = df[(df[fk] == row[fk]) & (df[ck] == row[ck])].sort_values("episode_index")
    before = same[same.episode_index < row.episode_index]["length"].sum()
    t0 = float(before) / fps
    return path, t0, t0 + float(row["length"]) / fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("episode", type=int)
    ap.add_argument("--cam", default="camera1")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0)
    ap.add_argument("--colors", type=int, default=128, help="palette size, 32 to 256. Lower is smaller.")
    ap.add_argument("--stats", default="diff", choices=["diff", "full"],
                    help="palettegen stats_mode. full samples whole frames, better when a small "
                         "colored object sits on a large static background.")
    ap.add_argument("--dither", default="bayer", choices=["bayer", "none"], help="none is much smaller: dither noise defeats inter-frame compression.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. install with:  conda install -c conda-forge ffmpeg")

    root = a.dataset if os.path.isdir(a.dataset) else os.path.join(CACHE, a.dataset)
    if not os.path.isdir(root):
        sys.exit(f"no such dataset: {root}")

    df = episode_meta(root)
    m = df[df.episode_index == a.episode]
    if m.empty:
        sys.exit(f"episode {a.episode} not in {a.dataset}; "
                 f"available {sorted(df.episode_index.tolist())}")

    path, t0, t1 = resolve_video(root, df, m.iloc[0], a.cam)
    t0 += a.start
    length = (t1 - t0) if a.dur <= 0 else min(a.dur, t1 - t0)
    if length <= 0:
        sys.exit(f"--start {a.start} is past the end of a {t1 - t0 + a.start:.1f}s episode")

    out = a.out or os.path.join("media", f"{os.path.basename(root)}_ep{a.episode}.gif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    palette = out + ".palette.png"

    vf = (f"setpts=PTS/{a.speed},fps={a.fps},"
          f"scale={a.width}:-1:flags=lanczos")

    dither = "dither=bayer:bayer_scale=3" if a.dither == "bayer" else "dither=none"

    base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{length:.3f}",
            "-i", path]
    try:
        subprocess.run(
            base + ["-vf", vf + f",palettegen=stats_mode={a.stats}:max_colors={a.colors}", palette],
            check=True)
        subprocess.run(
            base + ["-i", palette,
                    "-lavfi", vf + f" [x]; [x][1:v] paletteuse={dither}:diff_mode=rectangle",
                    "-loop", "0", out],
            check=True,
        )
    finally:
        if os.path.exists(palette):
            os.remove(palette)

    mb = os.path.getsize(out) / 1e6
    print(f"\nwrote {out}  ({mb:.1f} MB)")
    if mb > 10:
        print("over 10 MB, GitHub will be slow to render it. "
              "try --width 480, --fps 10, or a higher --speed")


if __name__ == "__main__":
    main()