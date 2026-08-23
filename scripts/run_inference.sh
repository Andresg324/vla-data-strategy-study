#!/usr/bin/env bash
# scripts/run_inference.sh
# Run one trained policy on one evaluation cell and record the rollouts.
#
# Usage:   bash scripts/run_inference.sh <policy> <cell>
# Example: bash scripts/run_inference.sh clean in_distribution

set -e

# Requires argument, print message and exit if missing
POLICY=${1:?usage: run_inference.sh <policy> <cell> [n_episodes]}
CELL=${2:?usage: run_inference.sh <policy> <cell> [n_episodes]}
NEPS=${3:-16}           # 15 scored episodes plus episode 0, the warm-up discarded by every analysis tool


# Ensures everything is spelt right before running, by checking policy and cell against options
case "$CELL" in
    in_distribution|new_positions|reduced_lighting|different_object|distractors|near_1in|near_2in|trained_t2) ;;
    *) echo "unknown cell '$CELL'"; exit 1 ;;
esac

case "$POLICY" in
    clean|randomized|recovery|color|color-slowpace|density|clean-seed2000|randomized-seed2000|color-seed2000|recovery-seed2000) ;;
    *) echo "unknown policy '$POLICY'"; exit 1 ;;
esac

FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/tty.usbmodem5B415324451}   # 12V arm, drives itself
HF_USER=${HF_USER:-your-hf-username}

# Confirm indices prior to running with tools/check_cameras.py
OVERHEAD_IDX=1
WRIST_IDX=0 

# chunk_size=50 and n_action_steps=50 (PROTOCOL.md §4.12) are inherited from the trained
# policy's config, not set here. Verify with: hf download <policy> config.json
# LeRobot appends a _YYYYMMDD_HHMMSS suffix to the repo_id at record time. Do not add one
# here; tools/rollout_paths.py requires exactly one and will ignore a name with two.

lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=my_follower_arm \
    --robot.cameras="{ camera1: {type: opencv, index_or_path: ${OVERHEAD_IDX}, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: ${WRIST_IDX}, width: 640, height: 480, fps: 30}}" \
    --policy.path=${HF_USER}/smolvla-cube-${POLICY} \
    --policy.device=${DEVICE:-mps} \
    --strategy.type=episodic \
    --strategy.reset_to_initial_position=true \
    --task="Pick up the cube and place it in the cup" \
    --dataset.repo_id=${HF_USER}/rollout_${POLICY}_${CELL} \
    --dataset.single_task="Pick up the cube and place it in the cup" \
    --dataset.fps=30 \
    --dataset.num_episodes=${NEPS} \
    --dataset.episode_time_s=45 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=${PUSH:-true} \
    --display_data=true
