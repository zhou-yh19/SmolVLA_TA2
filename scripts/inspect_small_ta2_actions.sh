#!/usr/bin/env bash
# Compare a trained SmolVLA2 checkpoint's action chunks against the recorded
# teleoperation actions, on episodes drawn at random from the TeleAvatar V2 data.
#
# This is the open-loop counterpart to the validation pass that now runs inside
# training: same comparison, but on a handful of samples you can actually read,
# with a CSV per chunk for plotting. It answers "does the policy reproduce the
# demonstrations", not "does the robot complete the task" -- see the header of
# scripts/inspect_checkpoint_actions.py for why those are different questions.
#
# Usage:
#   ./scripts/inspect_small_ta2_actions.sh                       # newest run, `last` checkpoint
#   CHECKPOINT=outputs/smolvla_ta2_multi_run06/checkpoints/010000/pretrained_model \
#       EPISODES=5 STARTS_PER_EPISODE=4 ./scripts/inspect_small_ta2_actions.sh
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/arapat/disk0/SmolVLA_TA2}"
CONDA_ENV="${CONDA_ENV:-vlab}"
EXP_NAME="${EXP_NAME:-smolvla_ta2_multi_run06}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${EXP_NAME}}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"

# `--dataset.root` is the PARENT directory, same as in the training script.
# Left empty to reuse whatever the checkpoint's train_config.json recorded.
DATASET_ROOT="${DATASET_ROOT:-/DATA/disk0/arapat/SmolVLA_TA2/datasets}"
DATASET_REPO_IDS="${DATASET_REPO_IDS:-small_lerobot_30fps}"

EPISODES="${EPISODES:-3}"
STARTS_PER_EPISODE="${STARTS_PER_EPISODE:-3}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda:0}"
INSPECT_OUTPUT_DIR="${INSPECT_OUTPUT_DIR:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# This host has no egress; everything needed is already on disk.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

if [[ ! -f "${CHECKPOINT}/model.safetensors" ]]; then
    echo "[error] No model.safetensors under ${CHECKPOINT}." >&2
    echo "[info] Available checkpoints:" >&2
    ls -1 "${OUTPUT_DIR}/checkpoints" 2>/dev/null | sed 's/^/  /' >&2 || true
    exit 1
fi
if [[ ! -f "${CHECKPOINT}/train_config.json" ]]; then
    echo "[error] ${CHECKPOINT}/train_config.json is missing; the dataset config cannot be recovered." >&2
    exit 1
fi

args=(
    --checkpoint "${CHECKPOINT}"
    --episodes "${EPISODES}"
    --starts-per-episode "${STARTS_PER_EPISODE}"
    --seed "${SEED}"
    --device "${DEVICE}"
)
if [[ -n "${DATASET_ROOT}" ]]; then
    args+=(--dataset-root "${DATASET_ROOT}")
fi
if [[ -n "${DATASET_REPO_IDS}" ]]; then
    args+=(--dataset-repo-ids "${DATASET_REPO_IDS}")
fi
if [[ -n "${INSPECT_OUTPUT_DIR}" ]]; then
    args+=(--output-dir "${INSPECT_OUTPUT_DIR}")
fi

echo "[info] checkpoint=${CHECKPOINT}"
echo "[info] episodes=${EPISODES} starts_per_episode=${STARTS_PER_EPISODE} seed=${SEED} device=${DEVICE}"

cd "${PROJECT_ROOT}"
exec python scripts/inspect_checkpoint_actions.py "${args[@]}"
