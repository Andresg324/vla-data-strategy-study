# tools/rollout_paths.py
# Shared resolution of timestamped rollout datasets.

import glob
import os
import re

CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser("~/.cache/huggingface/lerobot/your-hf-username"))
CELLS = ["in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors", "near_1in", "near_2in", "trained_t2"]
PATTERN = re.compile(r"^rollout_(?P<policy>.+?)_(?P<cell>" + "|".join(CELLS) + r")_(?P<stamp>\d{8}_\d{6})$")

def parse_policy(policy):
    for suffix, seed in (("-seed3000", 3000), ("-seed2000", 2000)):
        if policy.endswith(suffix):
            return policy[: -len(suffix)], seed
    return policy, 1000

def discover():
    # {(Policy, cell): path} for the latest dataset for each pair
    found, unmatched = {}, []
    for path in glob.glob(os.path.join(CACHE, "rollout_*")):
        if not os.path.isdir(path):
            continue
        m = PATTERN.match(os.path.basename(path))
        if not m:
            unmatched.append(os.path.basename(path))
            continue
        key = (m["policy"], m["cell"])
        if key not in found or m["stamp"] > found[key][0]:
            found[key] = (m["stamp"], path)
    if unmatched:
        print(f" note: {len(unmatched)} rollout_* directories did not match the naming pattern "
              f"and were ignored: {sorted(unmatched)[:5]}"
              + (" ..." if len(unmatched) > 5 else ""))
    return {k: v[1] for k, v in sorted(found.items())}

def resolve(policy, cell):
    # Path to the latest dataset for one (policy, cell)
    hit = discover().get((policy, cell))
    if hit is None:
        raise FileNotFoundError(f"No dataset for rollout_{policy}_{cell}_*")
    return hit