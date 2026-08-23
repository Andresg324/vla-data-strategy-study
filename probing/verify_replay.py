#!/usr/bin/env python3
"""
probing/verify_replay.py

Evidence for PROTOCOL.md §9: activations for probing are extracted by replaying saved
episodes from encoded video rather than from live camera frames. This replays one
reference frame through the trained policy five times and compares the predicted action
against the action recorded at that frame, so the replay discrepancy can be read against
the policy's own per-call sampling noise.

RUN: python probing/verify_replay.py
"""

import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

POLICY = "your-hf-username/smolvla-cube-clean"
REPO = "your-hf-username/rollout_clean_in_distribution_20260810_120038"
DEVICE = "mps"
TASK = "Pick up the cube and place it in the cup"

def main():
    policy = SmolVLAPolicy.from_pretrained(POLICY).to(DEVICE).eval()

    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=POLICY,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )

    ds = LeRobotDataset(REPO)
    frame = ds[0]
    print("frame keys:", sorted(frame.keys()))

    obs = {k: v.unsqueeze(0) for k, v in frame.items() if k.startswith("observation")}
    obs["task"] = [frame.get("task", TASK)]

    obs = pre(obs)
    print("after preprocess:", sorted(obs.keys()))

    policy.reset()
    with torch.no_grad():
        act = policy.select_action(obs)
    act = post(act)

    print("predicted:", act.squeeze().float().cpu().numpy().round(3))
    print("recorded :", frame["action"].numpy().round(3))

    # Check the Standard deviation and mean absolute error to understand difference between predicted and recorded
    def predict():
        f = ds[0]
        o = {k: v.unsqueeze(0) for k, v in f.items() if k.startswith("observation")}
        o["task"] = [f.get("task", TASK)]
        policy.reset()
        with torch.no_grad():
            return post(policy.select_action(pre(o))).squeeze().float().cpu().numpy()

    P = np.stack([predict() for _ in range(5)])
    rec = ds[0]["action"].numpy()

    np.set_printoptions(precision=3, suppress=True)
    print("per-call std   :", P.std(axis=0))
    print("mean pred      :", P.mean(axis=0))
    print("recorded       :", rec)
    print("mean abs error :", np.abs(P.mean(axis=0)-rec))

if __name__ == "__main__":
    main()
