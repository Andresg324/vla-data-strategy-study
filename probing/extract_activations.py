#!/usr/bin/env python3
"""
probing/extract_activations.py

Replay saved evaluation rollouts through a trained SmolVLA policy, capture one
layer's hidden activations, and write the .npz that probe_position.py and
probe_success.py consume.

Three modes:
  1. Find a layer to hook:
        python probing/extract_activations.py --policy clean --list-layers
  2. Extract for one policy across its eval cells:
        python probing/extract_activations.py --policy clean \
            --layer model.vlm_with_expert.lm_expert.norm --device mps --seed 0
  3. Merge the per-policy files into one:
        python probing/extract_activations.py --merge probing/out_np/activations_*.npz

CELLS and PATTERN must be kept in sync with tools/rollout_paths.py by hand.

The layer behind every reported result is model.vlm_with_expert.lm_expert.norm, the
action expert's final RMSNorm, 720 wide. Hook that module, never a LlamaDecoderLayer:
LeRobot's SmolVLA runs a custom interleaved forward that calls each layer's submodules
directly, so a hook on lm_expert.layers.N never fires, and neither --list-layers nor
get_module warns you, because the module does exist. The symptom is "0 activations".

SmolVLA samples action noise, so a hidden state is one draw rather than a fixed
function of the input. Two extractions of the same episode correlate at 0.998 on the
same device and 0.997 across CUDA and MPS, so device choice is irrelevant next to the
sampling. --seed seeds per (cell, episode) so a rerun reproduces, and so extracting a
subset of cells matches the same cells inside a full run. It cannot reproduce any run
made before seeding existed.
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import re

from lerobot.policies.factory import make_pre_post_processors


HF_USER = "your-hf-username"
CELLS = ["in_distribution", "new_positions", "reduced_lighting",
         "different_object", "distractors", "near_1in", "near_2in"]

TASK = "Pick up the cube and place it in the cup"  # This needs to be verbatim to the training
CACHE = os.environ.get("LEROBOT_CACHE", os.path.expanduser(f"~/.cache/huggingface/lerobot/{HF_USER}"))

PATTERN = re.compile(r"^rollout_(?P<policy>.+?)_(?P<cell>" + "|".join(CELLS) + r")_(?P<stamp>\d{8}_\d{6})$")

# ------------------------------------------------
# Imports that LeRobot moved between releases
# ------------------------------------------------

def _import_first(paths, attr):
    errors = []
    for path in paths:
        try:
            module = __import__(path, fromlist=[attr])
            return getattr(module, attr)
        except Exception as exc:
            errors.append(f" {path}: {exc}")
    raise ImportError(f"could not import {attr}:\n" + "\n".join(errors))

def load_policy(policy_slug, device):
    cls = _import_first(
        ["lerobot.policies.smolvla.modeling_smolvla",
         "lerobot.common.policies.smolvla.modeling_smolvla"],
         "SmolVLAPolicy",
    )
    repo = f"{HF_USER}/smolvla-cube-{policy_slug}"
    policy = cls.from_pretrained(repo)
    policy.to(device)
    policy.eval()
    
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=repo,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": device}}
    )

    return policy, pre, post

def load_dataset(repo_id):
    cls = _import_first(
        ["lerobot.datasets.lerobot_dataset",
         "lerobot.common.datasets.lerobot_dataset"],
         "LeRobotDataset",
    )
    return cls(repo_id)

# ------------------------------------
# Layer discovery and hooking
# ------------------------------------

def list_layers(policy):
    # Print every submodule name, pipe through grep to narrow it down
    for name, module in policy.named_modules():
        if name:
            print(f"{name:75s} {type(module).__name__}")


def get_module(policy, name):
    # Looks up module by the full dotted name
    modules = dict(policy.named_modules())
    if name not in modules:
        raise KeyError(f"layer '{name}' not found. Run --list-layers to see options.")
    return modules[name]

# One dict shared with the hook. The hook cannot return a value, so it stashes
# the activation here and the main loop reads it back out.
captured = {}

def hook_fn(module, inputs, output):
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    tensor = tensor.detach().float()
    if tensor.ndim == 3:
        # (batch, tokens, hidden), these are averaged over tokens to get one vector
        tensor = tensor.mean(dim=1)
    elif tensor.ndim > 3:
        tensor = tensor.flatten(2).mean(dim=2)
    captured["act"] = tensor[0].cpu().numpy() # batch size is always 1 (e.g., tensor[0])


# ----------------------------------------
# Data plumbing
# ----------------------------------------

def build_observation(item, device):
    # Turns one dataset row into the batch dict select_action() expects

    obs = {}
    for key, value in item.items():
        if key.startswith("observation.") and isinstance(value, torch.Tensor):
            obs[key] = value.unsqueeze(0).to(device) # adds the batch dimension
    obs["task"] = [item.get("task", TASK)]
    return obs


def episode_index_columns(ds):
    # Finds an episode ID for every row, without decoding the video frames
    
    try:
        return np.asarray(ds.hf_dataset["episode_index"])
    except Exception:
        return np.asarray([int(ds[i]["episode_index"]) for i in range(len(ds))])

def load_labels(paths):
    # {(condition, eval_cell, seed, episode): success} from the scored results
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return {(r.condition, r.eval_cell, int(r.seed), int(r.episode)): int(r.success) for r in df.itertuples()}

def parse_policy(policy):
    for suffix, seed in (("-seed3000", 3000), ("-seed2000", 2000)):
        if policy.endswith(suffix):
            return policy[: -len(suffix)], seed
    return policy, 1000

def discover():
    # {(Policy, cell): path} for the latest dataset for each pair
    found = {}
    for path in glob.glob(os.path.join(CACHE, "rollout_*")):
        m = PATTERN.match(os.path.basename(path))
        if not m or not os.path.isdir(path):
            continue
        key = (m["policy"], m["cell"])
        if key not in found or m["stamp"] > found[key][0]:
            found[key] = (m["stamp"], path)
    return {k: v[1] for k, v in sorted(found.items())}

# ----------------------------------------
# Extractions
# ----------------------------------------

_RUNS = None

def resolve_repo(policy_slug, cell):
    # Local CACHE directory name for the newest (policy, cell) rollout
    global _RUNS
    if _RUNS is None:
        _RUNS = discover()
    path = _RUNS.get((policy_slug, cell))
    if path is None:
        raise FileNotFoundError(f"No local dataset for rollout_{policy_slug}_{cell}_*")
    return f"{HF_USER}/{os.path.basename(path)}"

def extract_cell(policy, pre, policy_slug, condition, seed, cell, labels, device, repo_override=None, limit_episodes=None, next_uid=0, torch_seed=0):
    #Replays one (policy, cell) rollout dataset and returns parallel lists.

    #repo = repo_override or f"{HF_USER}/rollout_{condition}_{cell}"
    repo = repo_override or resolve_repo(policy_slug, cell)

    print(f" {repo}")
    ds = load_dataset(repo)
    ep_col = episode_index_columns(ds)

    out = {"X": [], "episode": [], "ep_true": [], "success": [], "t_from_end": []}
    episodes = [e for e in sorted({int(e) for e in ep_col}) if e != 0] #Not including the warm-up episode used
    if limit_episodes:
        episodes = episodes[:limit_episodes]


    for ep in episodes:
        rows = np.flatnonzero(ep_col == ep) # Already in frame order
        last = len(rows) - 1

        key = (condition, cell, seed, ep)
        if key not in labels:
            print(f" episode {ep}: no row in results, skipped")
            continue
        y = labels[key]

        policy.reset() # Clears the action queue so episodes don't bleed

        # Order is load-bearing: extract_cell() seeds on CELLS.index(cell), so reordering this
        # list changes every draw and breaks reproducibility against prior extractions.     
        
        # SmolVLA samples action noise, so hidden states are not reproducible without
        # this. Seed per (cell, episode) rather than once per run, so a draw does not
        # depend on which cells were extracted, or in what order.
        torch.manual_seed(torch_seed + 1000 * CELLS.index(cell) + ep)

        for t, row in enumerate(rows):
            obs = build_observation(ds[int(row)], device)
            obs = pre(obs)

            captured.pop("act", None) # Allows us to tell whether the hook fired
            with torch.no_grad():
                policy.select_action(obs)

                # SmolVLA predicts a chunk of actions, then pops from a queue for the next n steps
                # without running the model, so the hook fires at the chunk boundaries and this is
                # where we want to record; Forcing every pass would be a significant more compute and 
                # not how the policy behaves

                if "act" in captured:
                    out["X"].append(captured["act"])
                    out["episode"].append(next_uid)
                    out["ep_true"].append(int(ep))
                    out["success"].append(y)
                    out["t_from_end"].append(last - t)

        next_uid += 1

    print(f" {len(out['X'])} activations from {len(episodes)} episodes")
    return out, next_uid

def merge(paths, out_path):
    keys = ["X", "condition", "seed", "eval_cell", "ep_true", "success", "t_from_end"]
    parts = [np.load(p, allow_pickle=True) for p in paths]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}

    seeds = sorted({int(p["torch_seed"][0]) for p in parts if "torch_seed" in p})
    if seeds:
        merged["torch_seed"] = np.array(seeds)

    offset, eps = 0, []
    for p in parts:
        e = p["episode"].astype(int)
        eps.append(e + offset)
        offset += int(e.max()) + 1

    merged["episode"] = np.concatenate(eps)

    layers = sorted({str(p["layer"][0]) for p in parts if "layer" in p})
    if len(layers) > 1:
        raise SystemExit(f"refusing to merge across layer: {layers}")
    if layers:
        merged["layer"] = np.array(layers)

    np.savez(out_path, **merged)
    print(f"merged {len(paths)} files to {out_path} ({len(merged['X'])} rows), layer {layers[0] if layers else 'unknown'}")

# ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="clean | randomized | recovery | color")
    ap.add_argument("--layer", help="module name from --list-layers")
    ap.add_argument("--results", nargs="*", default=["documents/results_full.csv", "documents/exploratory.csv"])
    ap.add_argument("--device", default="mps", help="mps | cuda | cpu")
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--rollout-repo", help="override the dataset name (gate test)")
    ap.add_argument("--limit-episodes", type=int, help="Stop after N per cell")
    ap.add_argument("--outdir", default="probing/out_np")
    ap.add_argument("--list-layers", action="store_true")
    ap.add_argument("--merge", nargs="*")
    ap.add_argument("--seed", type=int, default=0, help="torch seed for the policy's action-noise sampling")
    args = ap.parse_args()

    if args.merge:
        os.makedirs(args.outdir, exist_ok=True)
        paths = [p for pat in args.merge for p in glob.glob(pat)]
        out = os.path.join(args.outdir, "activations_real.npz")
        paths = sorted(p for p in paths if os.path.abspath(p) != os.path.abspath(out))
        if not paths:
            raise SystemExit("no input files after excluding the merge target")
        merge(paths, out)
        return

    if not args.policy:
        ap.error("--policy is required unless --merge is given")

    policy, pre, _ = load_policy(args.policy, args.device)

    if args.list_layers:
        list_layers(policy)
        return

    condition, seed = parse_policy(args.policy)
    get_module(policy, args.layer).register_forward_hook(hook_fn)
    print(f"hooking layer: {args.layer}")
    labels = load_labels(args.results)

    X, cond, cells, eps, eptrue, succ, tfe = [], [], [], [], [], [], []
    uid = 0
    for cell in args.cells:
        try:
            part, uid = extract_cell(policy, pre, args.policy, condition, seed, cell, labels, args.device, args.rollout_repo, args.limit_episodes, uid, args.seed)
        except FileNotFoundError as e:
            print(f" skipping: {e}")
            continue
        n = len(part["X"])
        X.extend(part["X"])
        cond.extend([condition] * n)
        cells.extend([cell] * n)
        eps.extend(part["episode"])
        eptrue.extend(part["ep_true"])
        succ.extend(part["success"])
        tfe.extend(part["t_from_end"])

    if not X:
        raise SystemExit("no activations captured. Check --layer fired and that the episodes have rows in --results.")
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"activations_{args.policy}.npz")
    np.savez(
        out_path,
        X=np.stack(X),
        condition=np.array(cond),
        seed=np.full(len(X), seed, dtype=int),
        eval_cell=np.array(cells),
        episode=np.array(eps, dtype=int),
        ep_true=np.array(eptrue, dtype=int),
        success=np.array(succ, dtype=int),
        t_from_end=np.array(tfe, dtype=int),
        layer=np.array([args.layer]),
        torch_seed=np.array([args.seed]),
    )

    print(f"\nwrote {out_path} X = {np.stack(X).shape}")

if __name__ == "__main__":
    main()