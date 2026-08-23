#!/usr/bin/env python3
"""
tools/check_cameras.py
Saves one still frame from each study camera at the resolution the dataset
records (640x480), and measures the frame rate actually achieved.

This is a still capture, not a live preview: LeRobot's OpenCV build is
headless so cv2.imshow does not work. For a live view use
scripts/check_cameras_live.sh (rerun viewer).

Run from the repo root:
    conda activate lerobot && python tools/check_cameras.py
"""

import sys
import cv2
import time

# Role -> OpenCV index; keep this identical to record_dataset.sh
BACKEND = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
CAMERAS = {"overhead": 1, "wrist": 0}
WIDTH, HEIGHT, FPS = 640, 480, 30 # Same as the recording so that the previews match the experiments

WARMUP_S = 2.0 # Lets camera start streaming before a frame
MEASURE_S = 3.0 # Sample window for achieved frame rate

ok_all = True

for name, idx in CAMERAS.items():
    cap = cv2.VideoCapture(idx, BACKEND)
    if not cap.isOpened():
        print(f"{name}: could not open index {idx}")
        ok_all = False
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    good = None
    deadline = time.time() + WARMUP_S

    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            good = frame
        else:
            time.sleep(0.05)

    if good is None:
        print(f"{name}: opened index {idx} but never returned a frame")
        ok_all = False
        cap.release()
        continue

    count, start = 0, time.time()
    while time.time() - start < MEASURE_S:
        ok, frame = cap.read()
        if ok and frame is not None:
            good = frame
            count += 1

    measured = count / (time.time() - start)

    path = f"tools/preview_{name}.jpg"
    if not cv2.imwrite(path, good):
        print(f"{name}: Failed to write {path} (run from the repo root)")
        ok_all = False
        cap.release()
        continue

    h, w = good.shape[:2]
    claimed = cap.get(cv2.CAP_PROP_FPS)
    print(f"{name} (index {idx}): saved {path}")
    print(f" resolution {w} x {h} driver claims {claimed:.1f} fps, measured {measured:.1f} fps")

    if (w, h) != (WIDTH, HEIGHT):
        print(f" WARNING: {name} returned {w} x {h}, not the {WIDTH} x {HEIGHT} the protocol "
              f"assumes. Do not record until this is fixed.")
        ok_all = False

    if measured < 25:
        print(f" WARNING: {name} is running at {measured:.1f} fps, below the 30 fps the "
              f"protocol assumes. Do not record until this is fixed.")
        ok_all = False

    cap.release()

raise SystemExit(0 if ok_all else 1)