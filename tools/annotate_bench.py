#!/usr/bin/env python3
"""
tools/annotate_bench.py

Labels the apparatus photo for Figure A1. Positions are fractions of width and
height, so they survive a re-crop: (px, py) is the thing being labelled and
(tx, ty) is where the text sits. Run it, look at it, nudge the numbers, repeat.

RUN: python tools/annotate_bench.py media/bench_wide.jpeg
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

INK = "#0b0b0b"

# label, point x, point y, text x, text y
LABELS = [
    ("Overhead camera",  0.398, 0.133, 0.601, 0.075),
    ("Gantry",           0.196, 0.357, 0.071, 0.277),
    ("Cup",              0.331, 0.656, 0.220, 0.557),
    ("Clamp LED",        0.114, 0.789, 0.051, 0.627),
    ("Follower arm",     0.420, 0.708, 0.270, 0.827),
    ("Cube at T6",       0.459, 0.654, 0.562, 0.597),
    ("Leader arm",       0.658, 0.542, 0.630, 0.436),
    ("Clamp LED",        0.807, 0.570, 0.790, 0.476),
]

OUT = "figures/figure_A1_annotated.png"

def main(path, out=OUT):
    if not os.path.exists(path):
        raise SystemExit(f"no such image: {path}")
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 300, h / 300), dpi=300)
    ax.imshow(img)
    ax.axis("off")
    for text, px, py, tx, ty in LABELS:
        ax.annotate(text, xy=(px * w, py * h), xytext=(tx * w, ty * h),
                    fontsize=7, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.9, color=INK,
                                    shrinkA=2, shrinkB=2))
    fig.subplots_adjust(0, 0, 1, 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0)
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "media/bench_wide.jpeg",
         sys.argv[2] if len(sys.argv) > 2 else OUT)