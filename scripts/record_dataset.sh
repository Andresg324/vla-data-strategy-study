#!/usr/bin/env bash
# Record a teleoperated dataset for one condition of the study.
# Usage:  bash scripts/record_dataset.sh <condition> <num_episodes>
# Example: bash scripts/record_dataset.sh clean 50
set -e
CONDITION=${1:-clean} # Conditions are 'clean' | 'randomized' | 'recovery' | 'color'
NUM_EPISODES=${2:?usage: record_dataset.sh <condition> <num_episodes>}

# color-slowpace is deliberately absent. It was not a designed condition: it came from a
# recording session that unintentionally ran at a slower pace, was renamed afterwards, and is
# analyzed as exploratory only. run_inference.sh accepts it because the policy
# exists and was evaluated; it should not be recorded on purpose. density is different: it is
# an exploratory condition that was deliberately designed and recorded (PROTOCOL.md §8.30).

case "$CONDITION" in
    clean|randomized|recovery|color|density) ;;
    *) echo "unknown condition '$CONDITION' (clean|randomized|recovery|color|density)"; exit 1;;
esac

# ---- Hardware information ----
FOLLOWER_PORT=${FOLLOWER_PORT:-/dev/tty.usbmodem5B415324451}   # 12V arm that executes
LEADER_PORT=${LEADER_PORT:-/dev/tty.usbmodem5B415328441}       # 5V arm - moved manually
HF_USER=${HF_USER:-your-hf-username}

# Confirm indices with tools/check_cameras.py before every session
OVERHEAD_IDX=1
WRIST_IDX=0

# Two cameras -> dataset keys 'wrist' (Seeed idx0) and 'overhead' (C270 idx1)
# dataset name encodes the condition; instruction is the same across all conditions

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=my_follower_arm \
    --robot.cameras="{ overhead: {type: opencv, index_or_path: ${OVERHEAD_IDX}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_IDX}, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=${HF_USER}/cube-pickup-${CONDITION} \
    --dataset.single_task="Pick up the cube and place it in the cup" \
    --dataset.num_episodes=${NUM_EPISODES} \
    --dataset.fps=30 \
    --dataset.episode_time_s=45 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=${PUSH:-true}
