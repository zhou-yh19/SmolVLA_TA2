#!/usr/bin/env bash
#SBATCH -J smolvla_community_v2
#SBATCH -p gpu
#SBATCH --exclude=paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-15,paraai-n32-h-01-agent-25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH -t 72:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

# Paths and experiment settings. Values can be overridden with sbatch --export.
PROJECT_ROOT="${PROJECT_ROOT:-/home/bingxing2/home/scx7f0v/SmolVLA_TA2}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/home/bingxing2/home/scx7f0v/.conda/envs/vlaba100}"
DATASET_ROOT="${DATASET_ROOT:-/home/bingxing2/home/scx7f0v/LerobotData}"
COMMUNITY_SUBDIR="${COMMUNITY_SUBDIR:-community_dataset_v2}"
EXP_NAME="${EXP_NAME:-smolvla_community_v2_run01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${EXP_NAME}}"

# Leave DATASET_REPO_IDS empty to discover every LeRobot dataset below
# ${DATASET_ROOT}/${COMMUNITY_SUBDIR}. repo IDs must be relative to DATASET_ROOT
# and comma-separated, without a trailing /data component. Example:
# DATASET_REPO_IDS="community_dataset_v2/0x00raghu/toffee_red,community_dataset_v2/0x00raghu/toffee_blue,community_dataset_v2/1g0rrr/offline_dataset_name2,community_dataset_v2/1g0rrr/offline_dataset_name3"
DATASET_REPO_IDS="${DATASET_REPO_IDS:-}"
DATASET_SAMPLING_WEIGHTS="${DATASET_SAMPLING_WEIGHTS:-}"

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
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-community-v2}"
WANDB_NOTES="${WANDB_NOTES:-SmolVLA2 multi-dataset training on ${COMMUNITY_SUBDIR}}"

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
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${PROJECT_ROOT}"

if [[ -z "${DATASET_REPO_IDS}" ]]; then
    community_dir="${DATASET_ROOT%/}/${COMMUNITY_SUBDIR}"
    if [[ ! -d "${community_dir}" ]]; then
        echo "[error] Community dataset directory not found: ${community_dir}" >&2
        exit 2
    fi

    # A LeRobot dataset root is the parent of meta/info.json. The resulting
    # repo IDs are relative to DATASET_ROOT, as required by the dataset loader.
    mapfile -d '' dataset_info_files < <(
        find "${community_dir}" -type f -path '*/meta/info.json' -print0 | sort -z
    )
    dataset_repo_ids=()
    for info_file in "${dataset_info_files[@]}"; do
        dataset_dir="${info_file%/meta/info.json}"
        dataset_repo_ids+=("${dataset_dir#${DATASET_ROOT%/}/}")
    done
else
    IFS=',' read -r -a dataset_repo_ids <<< "${DATASET_REPO_IDS}"
fi

if (( ${#dataset_repo_ids[@]} < 2 )); then
    echo "[error] Multi-dataset training requires at least two datasets; found ${#dataset_repo_ids[@]}." >&2
    exit 2
fi

validated_repo_ids=()
for repo_id in "${dataset_repo_ids[@]}"; do
    # Accept an accidental /data suffix, but always pass the dataset root ID.
    repo_id="${repo_id#/}"
    repo_id="${repo_id%/}"
    repo_id="${repo_id%/data}"
    dataset_dir="${DATASET_ROOT%/}/${repo_id}"

    if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
        echo "[error] Invalid LeRobot dataset: ${dataset_dir}" >&2
        echo "[error] Expected metadata file: ${dataset_dir}/meta/info.json" >&2
        exit 2
    fi
    if [[ ! -d "${dataset_dir}/data" ]]; then
        echo "[error] Missing data directory: ${dataset_dir}/data" >&2
        exit 2
    fi
    first_parquet="$(find "${dataset_dir}/data" -type f -name '*.parquet' -print -quit)"
    if [[ -z "${first_parquet}" ]]; then
        echo "[error] No Parquet files found below ${dataset_dir}/data" >&2
        exit 2
    fi
    if [[ -d "${dataset_dir}/videos" ]]; then
        first_zero_byte_video="$(find "${dataset_dir}/videos" -type f -name '*.mp4' -size 0 -print -quit)"
        if [[ -n "${first_zero_byte_video}" ]]; then
            echo "[error] Found a zero-byte MP4 below ${dataset_dir}/videos" >&2
            find "${dataset_dir}/videos" -type f -name '*.mp4' -size 0 | sed -n '1,20p' >&2
            exit 3
        fi
    fi
    validated_repo_ids+=("${repo_id}")
done

DATASET_REPO_IDS="$(IFS=,; echo "${validated_repo_ids[*]}")"

if [[ -n "${DATASET_SAMPLING_WEIGHTS}" ]]; then
    IFS=',' read -r -a sampling_weights <<< "${DATASET_SAMPLING_WEIGHTS}"
    if (( ${#sampling_weights[@]} != ${#validated_repo_ids[@]} )); then
        echo "[error] DATASET_SAMPLING_WEIGHTS contains ${#sampling_weights[@]} weights, but ${#validated_repo_ids[@]} datasets were selected." >&2
        exit 2
    fi
fi

# Fail before launching distributed workers if CUDA or the PyAV runtime is broken.
EXPECTED_GPU_COUNT="${NUM_PROCESSES}" python - <<'PY'
import os

import av
import torch

expected = int(os.environ["EXPECTED_GPU_COUNT"])
actual = torch.cuda.device_count()
print(f"[info] PyTorch: {torch.__version__}; PyAV: {av.__version__}")
print(f"[info] CUDA available: {torch.cuda.is_available()}; visible GPUs: {actual}")
if not torch.cuda.is_available() or actual < expected:
    raise RuntimeError(f"Expected at least {expected} visible GPUs, got {actual}")
PY

train_args=(
    --policy.type=smolvla2
    --policy.vlm_model_name="${VLM_MODEL_NAME}"
    --policy.load_vlm_weights=true
    --policy.push_to_hub=false
    --policy.train_expert_only=false
    --policy.freeze_vision_encoder=false
    --dataset.repo_id="${DATASET_REPO_IDS}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --dataset.features_version=2
    --policy.max_action_dim=32
    --policy.max_state_dim=32
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
    --trackio.enable=false
)

if [[ -n "${DATASET_SAMPLING_WEIGHTS}" ]]; then
    train_args+=(--dataset.sampling_weights="${DATASET_SAMPLING_WEIGHTS}")
fi
if [[ "${WANDB_ENABLED}" == "1" ]]; then
    train_args+=(--wandb.enable=true)
else
    train_args+=(--wandb.enable=false)
fi

echo "[info] experiment=${EXP_NAME}"
echo "[info] dataset_root=${DATASET_ROOT} datasets=${#validated_repo_ids[@]}"
for i in "${!validated_repo_ids[@]}"; do
    printf '[info] dataset[%d]=%s\n' "${i}" "${validated_repo_ids[i]}"
done
echo "[info] GPUs=${NUM_PROCESSES} batch_per_gpu=${BATCH_SIZE} global_batch=$((NUM_PROCESSES * BATCH_SIZE))"
echo "[info] workers_per_process=${NUM_WORKERS} steps=${NUM_TRAIN_STEPS} output=${OUTPUT_DIR}"
echo "[info] scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS} scheduler_decay_steps=${SCHEDULER_DECAY_STEPS} scheduler_decay_lr=${SCHEDULER_DECAY_LR}"

accelerate launch \
    --config_file "${PROJECT_ROOT}/accelerate_configs/multi_gpu.yaml" \
    --num_processes "${NUM_PROCESSES}" \
    "${PROJECT_ROOT}/src/lerobot/scripts/train.py" \
    "${train_args[@]}"

echo "[info] Training completed: ${OUTPUT_DIR}"
