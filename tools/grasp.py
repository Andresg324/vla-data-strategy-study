"""
Shared grasp-event detection for rollout and demonstration action streams.
"""

import numpy as np

GRIPPER = 5                 # index of the gripper in the 6-dim commanded action vector
OPEN_THR = 15.0             # gripper command above this means the jaw is open

def _runs(mask):
    # Gives (start, end) index pairs for each contiguous True run
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return list(zip(starts, ends))

def _merge(runs, max_gap=3):
    # A single physical opening can dip below threshold for a couple of frames;
    # This stitches those together before filtering the length
    out = []
    for s, e in runs:
        if out and s - out[-1][1] <= max_gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out

def grasp_pose(A, min_run=5, max_gap=3):
    """Pose at the moment the gripper begins closing out of its approach-open.

    Robust to the gripper stalling on the cube rather than reaching the fully
    closed threshold. Returns (pose, frame, released), where released is True
    if a second sustained open follows, i.e. the policy let go of something.

    min_run=5 translates to roughly 167 ms at 30fps, and max_gap = 3 is 100ms
    """

    g = A[:, GRIPPER]
    opens = [r for r in _merge(_runs(g > OPEN_THR), max_gap) if r[1] - r[0] >= min_run]

    if not opens:
        return None, None, None

    start, end = opens[0]
    if end >= len(g):
        return None, None, None         # Still open on the last frame (e.g., no close event)

    i = int(end) - 1                    # last frame of the approach is still open
    return A[i, :GRIPPER], i, len(opens) >= 2
