#!/usr/bin/env bash
# scripts/preview_cameras.sh
# Fast live preview of both cameras, addressed BY NAME rather than by index, so it
# survives the macOS index shuffle. Use this to confirm the cameras are alive and framed;
# use check_cameras_live.sh when you also need teleoperation running.
# Overhead = Logitech C270 (gantry). Wrist = Seeed "Web Camera".
# Quit: Ctrl-C, or close both windows.

# macOS only: -f avfoundation and the device names below are Darwin specific. On Linux use
# -f v4l2 -i /dev/video0 and /dev/video1, and confirm which index is which with
# tools/check_cameras.py.

set -e
trap 'kill 0' EXIT INT TERM

command -v ffplay >/dev/null || { echo "ffplay not found: conda install -c conda-forge ffmpeg"; exit 1; }

ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "C270 HD WEBCAM" &
ffplay -f avfoundation -framerate 30 -video_size 640x480 -i "Web Camera" &

wait