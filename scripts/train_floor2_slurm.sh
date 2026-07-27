#!/usr/bin/env bash
#SBATCH -J smolvla_floor2
#SBATCH -p gpu
#SBATCH --exclude=paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-15,paraai-n32-h-01-agent-25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# Paths and experiment settings. Every value can be overridden with sbatch --export.
PROJECT_ROOT="${PROJECT_ROOT:-/home/bingxing2/home/scx7f0v/SmolVLA_TA2}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/bingxing2/home/scx7f0v/.conda/envs/vlab}"
DATASET_ROOT="${DATASET_ROOT:-/home/bingxing2/home/scx7f0v/LerobotData}"
DATASET_REPO_ID="${DATASET_REPO_ID:-floor2}"
EXP_NAME="${EXP_NAME:-smolvla_floor2_run01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${EXP_NAME}}"
POLICY_PATH="${POLICY_PATH:-}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-80000}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-0}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:--1}"
USE_AMP="${USE_AMP:-true}"

VLM_MODEL_NAME="${VLM_MODEL_NAME:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-floor2}"
WANDB_NOTES="${WANDB_NOTES:-SmolVLA2 training on ${DATASET_REPO_ID}}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
CLEAR_PROXY="${CLEAR_PROXY:-0}"

module purge >/dev/null 2>&1 || true
module load miniforge3/24.1
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV_PREFIX}"
set -u

# This ARM cluster otherwise resolves /usr/lib64/libstdc++.so.6 first, which
# makes PyAV/OpenVINO fail with "GLIBCXX_3.4.26 not found".
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if [[ "${CLEAR_PROXY}" == "1" ]]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

cd "${PROJECT_ROOT}"

DATASET_DIR="${DATASET_ROOT%/}/${DATASET_REPO_ID}"
if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
    echo "[error] LeRobot dataset not found: ${DATASET_DIR}" >&2
    echo "[error] Expected metadata file: ${DATASET_DIR}/meta/info.json" >&2
    exit 2
fi

if [[ ! -f "${DATASET_DIR}/meta/teleavatar_v2_stats.json" ]]; then
    echo "[error] Missing adapted TeleAvatar V2 statistics:" >&2
    echo "[error] ${DATASET_DIR}/meta/teleavatar_v2_stats.json" >&2
    echo "[info] Compute them before submitting this GPU job:" >&2
    echo "  cd ${PROJECT_ROOT}" >&2
    echo "  python scripts/compute_teleavatar_v2_stats.py --dataset ${DATASET_DIR}" >&2
    exit 4
fi

if [[ -d "${DATASET_DIR}/videos" ]]; then
    zero_byte_videos="$(find "${DATASET_DIR}/videos" -type f -name '*.mp4' -size 0 | wc -l)"
    if (( zero_byte_videos > 0 )); then
        echo "[error] Found ${zero_byte_videos} zero-byte MP4 files under ${DATASET_DIR}/videos." >&2
        find "${DATASET_DIR}/videos" -type f -name '*.mp4' -size 0 | sed -n '1,20p' >&2
        exit 3
    fi
fi

# Fail before launching distributed workers if CUDA or the PyAV runtime is broken.
EXPECTED_GPU_COUNT="${NUM_PROCESSES}" python - <<'PY'
import os

import torch
import av

expected = int(os.environ["EXPECTED_GPU_COUNT"])
actual = torch.cuda.device_count()
compiled_arches = set(torch.cuda.get_arch_list())
print(f"[info] PyTorch: {torch.__version__}; PyAV: {av.__version__}")
print(f"[info] CUDA available: {torch.cuda.is_available()}; visible GPUs: {actual}")
if not torch.cuda.is_available() or actual < expected:
    raise RuntimeError(f"Expected at least {expected} visible GPUs, got {actual}")

unsupported = []
for index in range(actual):
    major, minor = torch.cuda.get_device_capability(index)
    required_arch = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name(index)
    if required_arch not in compiled_arches:
        unsupported.append(f"GPU {index}: {name} requires {required_arch}")
if unsupported:
    raise RuntimeError(
        "This PyTorch build supports "
        f"{sorted(compiled_arches)}, but the allocated GPUs are unsupported: "
        + "; ".join(unsupported)
    )
PY

train_args=(
    --policy.vlm_model_name="${VLM_MODEL_NAME}"
    --policy.push_to_hub=false
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --output_dir="${OUTPUT_DIR}"
    --job_name="${EXP_NAME}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --steps="${NUM_TRAIN_STEPS}"
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --policy.use_amp="${USE_AMP}"
    --wandb.project="${WANDB_PROJECT}"
    --wandb.notes="${WANDB_NOTES}"
    --wandb.mode="${WANDB_MODE}"
    --wandb.disable_artifact="${WANDB_DISABLE_ARTIFACT}"
    --trackio.enable=false
)

if [[ -n "${POLICY_PATH}" ]]; then
    train_args+=(
        --policy.path="${POLICY_PATH}"
        --policy.load_vlm_weights=false
    )
    policy_init="pretrained:${POLICY_PATH}"
else
    train_args+=(
        --policy.type=smolvla2
        --policy.load_vlm_weights=true
    )
    policy_init="scratch:smolvla2"
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
    train_args+=(--wandb.enable=true)
else
    train_args+=(--wandb.enable=false)
fi

echo "[info] experiment=${EXP_NAME} dataset=${DATASET_DIR}"
echo "[info] policy_init=${policy_init} optimizer_lr=${OPTIMIZER_LR} train_expert_only=${TRAIN_EXPERT_ONLY} freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
echo "[info] GPUs=${NUM_PROCESSES} batch_per_gpu=${BATCH_SIZE} global_batch=$((NUM_PROCESSES * BATCH_SIZE))"
echo "[info] workers_per_process=${NUM_WORKERS} steps=${NUM_TRAIN_STEPS} output=${OUTPUT_DIR}"
echo "[info] scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS} scheduler_decay_steps=${SCHEDULER_DECAY_STEPS} scheduler_decay_lr=${SCHEDULER_DECAY_LR}"
echo "[info] wandb_enabled=${WANDB_ENABLED} wandb_mode=${WANDB_MODE}"

accelerate launch \
    --config_file "${PROJECT_ROOT}/accelerate_configs/multi_gpu.yaml" \
    --num_processes "${NUM_PROCESSES}" \
    "${PROJECT_ROOT}/src/lerobot/scripts/train.py" \
    "${train_args[@]}"

echo "[info] Training completed: ${OUTPUT_DIR}"
