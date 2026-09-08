#!/usr/bin/env bash
# Fine-tune SmolVLA2 on the TeleAvatar V2 datasets across four local GPUs.
#
# Derived from scripts/train_floor2_slurm.sh, with the SLURM parts removed.
# Multi-dataset handling follows scripts/train_community_v2_multidataset_slurm.sh:
# a comma-separated --dataset.repo_id makes factory.py build a MultiLeRobotDataset.
#
# The first run (small only, 5000 steps, scratch action expert) produced a
# policy that only jittered in place. The action expert was randomly
# initialised and 4695 frames cannot train one from scratch, so POLICY_PATH now
# defaults to the pretrained smolvla_robotwin checkpoint.
#
# Every value can be overridden from the environment:
#   DATASET_REPO_IDS=small_lerobot_30fps NUM_TRAIN_STEPS=2000 \
#       ./scripts/train_small_ta2_local.sh
#
# To continue an interrupted run from its newest checkpoint, restoring the
# optimizer, the scheduler and the step counter:
#   RESUME=1 ./scripts/train_small_ta2_local.sh
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/arapat/disk0/SmolVLA_TA2}"
CONDA_ENV="${CONDA_ENV:-vlab}"

# wandb and huggingface both default to caches under $HOME. Here $HOME is a
# 435 GB partition already near capacity while the project sits on a 14 TB
# volume, so every large cache is redirected onto the project. run06 died at
# step 6000 because wandb staged a 1.8 GB copy of the checkpoint into
# ~/.local/share/wandb and $HOME had no room left.
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${PROJECT_ROOT}/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.wandb_cache}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_home}"

# `--dataset.root` is the PARENT directory: make_dataset joins root/repo_id.
DATASET_ROOT="${DATASET_ROOT:-/DATA/disk0/arapat/SmolVLA_TA2/datasets}"

# Comma-separated. factory.py splits on "," and builds a MultiLeRobotDataset
# when more than one ID is given; a single ID still takes the LeRobotDataset
# path, so this variable covers both cases. All three are robot_type
# "teleavatar", so MultiLeRobotDataset groups their statistics under one key
# and policies/factory.py flattens that group back to per-feature stats.
DATASET_REPO_IDS="${DATASET_REPO_IDS:-small_lerobot_30fps,middle_lerobot_30fps,large_lerobot_30fps}"

# Optional per-dataset sampling weights, comma-separated, same order and count
# as DATASET_REPO_IDS. Empty means uniform sampling over the concatenation,
# so the largest dataset dominates in proportion to its length.
DATASET_SAMPLING_WEIGHTS="${DATASET_SAMPLING_WEIGHTS:-}"

EXP_NAME="${EXP_NAME:-smolvla_ta2_multi_run07}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${EXP_NAME}}"

# RESUME=1 continues an existing run instead of starting a new fine-tune from
# POLICY_PATH. The difference matters: POLICY_PATH loads weights only and
# restarts the step counter, the optimizer moments and the LR schedule from
# zero, whereas --resume=true restores all four, so the run picks up exactly
# where it stopped. The two are mutually exclusive by design
# (configs/train.py:85), and this script wires up whichever one applies.
#
# RESUME_CHECKPOINT defaults to the `last` symlink that update_last_checkpoint
# maintains (train.py:549). Override it to rewind to an earlier step:
#   RESUME=1 RESUME_CHECKPOINT=outputs/<exp>/checkpoints/004000 \
#       ./scripts/train_small_ta2_local.sh
RESUME="${RESUME:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last}"

# The complete pretrained SmolVLA checkpoint to fine-tune from: backbone AND
# action expert. Setting it switches the run to --policy.path below.
#
# Do NOT clear this to get the old behaviour by accident. An empty POLICY_PATH
# means --policy.type=smolvla2, which loads pretrained SmolVLM weights for the
# backbone but leaves the action expert randomly initialised -- that is what
# run03 did, and its policy only jittered. smolvla_robotwin is preferred over
# smolvla_base here because it was trained on bimanual data.
POLICY_REPO_ID="${POLICY_REPO_ID:-lerobot/smolvla_robotwin}"
POLICY_LOCAL_DIR="${POLICY_LOCAL_DIR:-${PROJECT_ROOT}/weights/${POLICY_REPO_ID}}"
if [[ -z "${POLICY_PATH:-}" && -f "${POLICY_LOCAL_DIR}/config.json" ]]; then
    POLICY_PATH="${POLICY_LOCAL_DIR}"
fi
POLICY_PATH="${POLICY_PATH:-${POLICY_REPO_ID}}"

# GPUs to use. BATCH_SIZE is PER PROCESS, so the global batch is
# NUM_PROCESSES * BATCH_SIZE. Pinned to 4 because BATCH_SIZE, OPTIMIZER_LR and
# NUM_TRAIN_STEPS below are all sized for a global batch of 32; "auto" would
# grab all 8 GPUs on this host and silently double the global batch, leaving
# the LR and step budget mismatched. Set NUM_PROCESSES=auto for every GPU, but
# re-check those three values if you do.
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# 20000 steps over the combined small+middle+large datasets. The old 5000-step
# budget was sized for 4695 frames alone; the combined set is several times
# that, so the same step count would be only a few epochs. Fine-tuning a
# pretrained expert also tolerates a longer run than training one from scratch.
# The exact epoch count is printed at startup once the frame totals are known.
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-20000}"
# Per-process batch. Sized for 80 GB A100s: 4 x 8 gives a global batch of 32.
# Three 512x512 cameras and chunk_size=50 dominate activation memory.
BATCH_SIZE="${BATCH_SIZE:-16}"
# Per process, so 4 x 8 = 32 loader workers decoding large MP4 streams.
NUM_WORKERS="${NUM_WORKERS:-4}"
# Warmup matters more at a larger batch and a higher peak LR. ~5% of the run.
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${NUM_TRAIN_STEPS}}"
# Cosine floor. The config hardcodes 2.5e-6 against a 2.5e-5 peak, so keep the
# same 1:10 ratio against the peak LR set below.
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-5}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
EVAL_FREQ="${EVAL_FREQ:--1}"

# Validation holdout, in whole episodes per dataset. Whole episodes because
# neighbouring frames at 30 fps are near-duplicates, so a frame-level split
# would score the model on data it has effectively trained on. Set to 0 to
# train on everything. With 135 episodes across the TA2 datasets, 3 per dataset
# is a few percent of the data -- enough to see a train/val gap open up,
# cheap enough not to matter for fitting.
VAL_EPISODES_PER_DATASET="${VAL_EPISODES_PER_DATASET:-3}"
# The split is a deterministic function of this seed and each dataset's
# repo_id. Keep it fixed across a comparison; changing it reshuffles the
# holdout and invalidates every earlier val number.
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-1000}"
# Validation runs on the main process at every checkpoint. VAL_MAX_BATCHES
# batches of denoising loss, of which the first VAL_ACTION_BATCHES also get a
# full 10-step open-loop rollout for the per-joint error in radians.
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-50}"
VAL_ACTION_BATCHES="${VAL_ACTION_BATCHES:-10}"

# The VLM backbone. Prefer the copy under weights/ so the run needs no network:
# transformers' from_pretrained accepts a local directory, and every call site
# (modeling_smolvla2.py:486 and smolvlm_with_expert2.py:93/101/103) reads this
# same value. Falls back to the Hub id when the local copy is absent.
VLM_REPO_ID="${VLM_REPO_ID:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
VLM_LOCAL_DIR="${VLM_LOCAL_DIR:-${PROJECT_ROOT}/weights/${VLM_REPO_ID}}"
if [[ -z "${VLM_MODEL_NAME:-}" && -f "${VLM_LOCAL_DIR}/config.json" ]]; then
    VLM_MODEL_NAME="${VLM_LOCAL_DIR}"
fi
VLM_MODEL_NAME="${VLM_MODEL_NAME:-${VLM_REPO_ID}}"
# The upstream 2.5e-5 was tuned at batch 2. Global batch is now 32 (16x), so
# sqrt-scaling gives 2.5e-5 * 4 = 1e-4. That was the right peak for run03,
# which trained an action expert from scratch. This run fine-tunes pretrained
# weights instead, where too high a peak erases what the checkpoint already
# knows, so the peak is halved to 5e-5.
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-false}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-true}"
USE_AMP="${USE_AMP:-true}"

# On by default: offline mode needs no network and writes structured metrics
# under wandb/, which is the only record that survives. The plain log stream
# is easy to lose (four ranks interleave into one file).
WANDB_ENABLED="${WANDB_ENABLED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-ta2}"
WANDB_MODE="${WANDB_MODE:-offline}"
# log_policy (wandb_utils.py:134) copies every checkpoint's model.safetensors
# into wandb's artifact staging directory. Offline runs never upload, so those
# 1.8 GB copies accumulate until the partition fills; that is what killed run06
# at step 6000, after the checkpoint itself had already been written. With this
# at 1, log_policy returns on its first line (wandb_utils.py:127). Set it to 0
# only when WANDB_MODE=online and the artifacts are actually wanted.
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-1}"
# Only consulted when RESUME=1, and only as an override: the run id is normally
# recovered from ${OUTPUT_DIR}/wandb. Set it when several runs share an output
# directory, or to attach the resumed run to a specific earlier one.
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# This host has no egress. Without these, huggingface_hub burns ~2 minutes on
# retries before failing. Set HF_HUB_OFFLINE=0 to allow downloads.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${PROJECT_ROOT}"
mkdir -p logs

# Resolved here, not at declaration time: torch is only importable once the
# conda environment is active.
if [[ "${NUM_PROCESSES}" == "auto" ]]; then
    NUM_PROCESSES="$(python -c "import torch; print(torch.cuda.device_count())")"
    echo "[warn] NUM_PROCESSES=auto resolved to ${NUM_PROCESSES}; global batch is" >&2
    echo "[warn] now ${NUM_PROCESSES} x ${BATCH_SIZE}. Verify OPTIMIZER_LR and NUM_TRAIN_STEPS." >&2
    if [[ -z "${NUM_PROCESSES}" || "${NUM_PROCESSES}" == "0" ]]; then
        echo "[error] No CUDA device detected." >&2
        exit 1
    fi
fi

# --- Preflight (mirrors the cluster script's checks) -----------------------
# Every dataset is checked, not just the first. The adapted-stats sidecar in
# particular is per dataset copy: it is generated locally and does not travel
# with an rsync of the repository, so a newly copied dataset will be missing it.
IFS=',' read -r -a dataset_repo_ids <<< "${DATASET_REPO_IDS}"
if (( ${#dataset_repo_ids[@]} == 0 )); then
    echo "[error] DATASET_REPO_IDS is empty." >&2
    exit 2
fi

validated_repo_ids=()
missing_stats=()
total_frames=0
total_episodes=0

for repo_id in "${dataset_repo_ids[@]}"; do
    # Tolerate stray whitespace and an accidental /data suffix.
    repo_id="${repo_id#"${repo_id%%[![:space:]]*}"}"
    repo_id="${repo_id%"${repo_id##*[![:space:]]}"}"
    repo_id="${repo_id#/}"
    repo_id="${repo_id%/}"
    repo_id="${repo_id%/data}"
    [[ -z "${repo_id}" ]] && continue
    dataset_dir="${DATASET_ROOT%/}/${repo_id}"

    if [[ ! -f "${dataset_dir}/meta/info.json" ]]; then
        echo "[error] LeRobot dataset not found: ${dataset_dir}" >&2
        echo "[info] Datasets present under ${DATASET_ROOT}:" >&2
        find "${DATASET_ROOT%/}" -maxdepth 3 -name info.json -path '*/meta/*' \
            -print 2>/dev/null | sed 's|/meta/info.json$||; s|^|  |' >&2
        exit 2
    fi

    # robot_type selects the adapter. Without "teleavatar" the raw 72-dim
# state/action reaches the policy instead of the adapted 16/16.
    # Captured rather than read from a process substitution: with `read`, a
    # python failure still returns success and leaves robot_type empty.
    info_line="$(python - "${dataset_dir}/meta/info.json" <<'INFO'
import json, sys
info = json.load(open(sys.argv[1]))
print(info.get("robot_type", ""), info.get("total_frames", 0) or 0, info.get("total_episodes", 0) or 0)
INFO
)" || { echo "[error] Could not parse ${dataset_dir}/meta/info.json" >&2; exit 2; }
    read -r robot_type frames episodes <<< "${info_line}"

    if [[ "${robot_type}" != "teleavatar" ]]; then
        echo "[error] ${repo_id}: robot_type is '${robot_type}', expected 'teleavatar'." >&2
        echo "[error] The adapter is selected by robot_type; without it the raw" >&2
        echo "[error] 72-dim state/action would be fed to the policy." >&2
        exit 2
    fi

    if [[ ! -f "${dataset_dir}/meta/teleavatar_v2_stats.json" ]]; then
        missing_stats+=("${dataset_dir}")
    elif ! python - "${dataset_dir}/meta/teleavatar_v2_stats.json" <<'STATS'
import json, sys
from lerobot.datasets.adapters.teleavatar import TeleavatarV2Adapter
TeleavatarV2Adapter(adapted_stats=json.load(open(sys.argv[1])))._require_adapted_stats()
STATS
    then
        missing_stats+=("${dataset_dir}")
    fi

    if [[ -d "${dataset_dir}/videos" ]]; then
        first_zero_byte="$(find "${dataset_dir}/videos" -type f -name '*.mp4' -size 0 -print -quit)"
        if [[ -n "${first_zero_byte}" ]]; then
            echo "[error] Found zero-byte MP4 files under ${dataset_dir}/videos:" >&2
            find "${dataset_dir}/videos" -type f -name '*.mp4' -size 0 | sed -n '1,20p' >&2
            exit 3
        fi
    fi

    echo "[info] dataset ${repo_id}: ${episodes} episodes, ${frames} frames"
    total_frames=$(( total_frames + frames ))
    total_episodes=$(( total_episodes + episodes ))
    validated_repo_ids+=("${repo_id}")
done

# Reported together so one run of the stats script fixes every dataset at once.
if (( ${#missing_stats[@]} > 0 )); then
    echo "[error] Missing or stale TeleAvatar V2 statistics for ${#missing_stats[@]} dataset(s)." >&2
    echo "[info] The gripper effort-to-trigger transform is piecewise, so raw" >&2
    echo "[info] LeRobot stats cannot be sliced. Generate them:" >&2
    for dataset_dir in "${missing_stats[@]+"${missing_stats[@]}"}"; do
        echo "  python scripts/compute_teleavatar_v2_stats.py --dataset ${dataset_dir}" >&2
    done
    exit 4
fi

DATASET_REPO_IDS="$(IFS=,; echo "${validated_repo_ids[*]}")"
num_datasets="${#validated_repo_ids[@]}"

if [[ -n "${DATASET_SAMPLING_WEIGHTS}" ]]; then
    IFS=',' read -r -a sampling_weights <<< "${DATASET_SAMPLING_WEIGHTS}"
    if (( ${#sampling_weights[@]} != num_datasets )); then
        echo "[error] DATASET_SAMPLING_WEIGHTS has ${#sampling_weights[@]} entries but ${num_datasets} datasets were selected." >&2
        exit 2
    fi
fi

# Resolving the VLM backbone requires either a local directory or network
# access. Offline runs otherwise fail only after the dataset is built.
if [[ "${VLM_MODEL_NAME}" == */* && -d "${VLM_MODEL_NAME}" ]]; then
    for required in config.json preprocessor_config.json tokenizer_config.json; do
        if [[ ! -f "${VLM_MODEL_NAME}/${required}" ]]; then
            echo "[error] Backbone directory is incomplete: ${VLM_MODEL_NAME}" >&2
            echo "[error] Missing ${required}" >&2
            exit 5
        fi
    done
    echo "[info] VLM backbone: local dir ${VLM_MODEL_NAME}"
elif [[ "${HF_HUB_OFFLINE}" == "1" ]]; then
    echo "[error] VLM_MODEL_NAME=${VLM_MODEL_NAME} is a Hub id, but this run is offline" >&2
    echo "[error] and no local copy exists at:" >&2
    echo "[error]   ${VLM_LOCAL_DIR}" >&2
    echo "[info] Download it on a networked machine and copy it there:" >&2
    echo "  hf download ${VLM_REPO_ID} --local-dir weights/${VLM_REPO_ID}" >&2
    echo "[info] Or allow downloads for this run with HF_HUB_OFFLINE=0." >&2
    exit 5
else
    echo "[info] VLM backbone: Hub id ${VLM_MODEL_NAME} (downloads enabled)"
fi

# Same failure mode for the policy checkpoint: an offline run that names a Hub
# id it cannot fetch dies only after the datasets are built, minutes in.
if [[ "${RESUME}" == "1" ]]; then
    # POLICY_PATH is not consulted at all on a resume, so it is not validated
    # either. Everything the run needs comes out of the checkpoint: the config
    # from pretrained_model/, the weights and optimizer state from
    # training_state/.
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT%/}"
    resume_config="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
    resume_state="${RESUME_CHECKPOINT}/training_state/training_step.json"

    if [[ ! -f "${resume_config}" ]]; then
        echo "[error] RESUME=1 but there is no checkpoint config at:" >&2
        echo "[error]   ${resume_config}" >&2
        echo "[info] Checkpoints present under ${OUTPUT_DIR}/checkpoints:" >&2
        find "${OUTPUT_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -print 2>/dev/null \
            | sort | sed 's|^|  |' >&2
        echo "[info] Drop RESUME=1 to start a fresh run from POLICY_PATH." >&2
        exit 6
    fi

    # A pretrained_model/ without its sibling training_state/ can only be
    # fine-tuned from step 0 via POLICY_PATH; load_training_state
    # (train.py:393) needs the optimizer, scheduler and step files.
    if [[ ! -f "${resume_state}" ]]; then
        echo "[error] Checkpoint has no training state: ${resume_state}" >&2
        echo "[error] Resuming needs training_state/ alongside pretrained_model/." >&2
        echo "[info] To fine-tune from these weights instead, starting the step" >&2
        echo "[info] counter and the LR schedule over:" >&2
        echo "  POLICY_PATH=${RESUME_CHECKPOINT}/pretrained_model ./scripts/train_small_ta2_local.sh" >&2
        exit 6
    fi

    resume_step="$(python - "${resume_state}" <<'STEP'
import json, sys
print(int(json.load(open(sys.argv[1]))["step"]))
STEP
)" || { echo "[error] Could not read the step from ${resume_state}" >&2; exit 6; }

    # The training loop runs `for step in range(resume_step, cfg.steps)`, so a
    # budget at or below the restored step exits immediately after writing one
    # checkpoint, which reads like a successful run that trained nothing.
    if (( resume_step >= NUM_TRAIN_STEPS )); then
        echo "[error] Checkpoint is already at step ${resume_step}, but" >&2
        echo "[error] NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS}. There is nothing left to run." >&2
        echo "[info] Raise the budget to continue past it, e.g.:" >&2
        echo "  RESUME=1 NUM_TRAIN_STEPS=$(( resume_step + 10000 )) ./scripts/train_small_ta2_local.sh" >&2
        exit 6
    fi

    echo "[info] resume: ${RESUME_CHECKPOINT}"
    echo "[info] resume: step ${resume_step} -> ${NUM_TRAIN_STEPS} ($(( NUM_TRAIN_STEPS - resume_step )) steps remaining)"

    # WandBLogger needs the original run id whenever cfg.resume is set: it calls
    # wandb.init(resume="must"), and when the config carries run_id=null -- which
    # a checkpoint's train_config.json does -- it falls back to globbing
    # wandb/latest-run/run-*.wandb (wandb_utils.py:52). That symlink is written
    # on a clean shutdown, so a crashed run leaves it missing or stale and the
    # resume dies before the first step. Resolve the id from the run directories
    # themselves instead, newest first, and pass it explicitly.
    resume_wandb_run_id="${WANDB_RUN_ID}"
    if [[ "${WANDB_ENABLED}" == "1" ]]; then
        wandb_run_files=()
        if [[ -z "${resume_wandb_run_id}" ]]; then
            while IFS= read -r wandb_run_file; do
                [[ -n "${wandb_run_file}" ]] && wandb_run_files+=("${wandb_run_file}")
            done < <(ls -1t "${OUTPUT_DIR}"/wandb/*/run-*.wandb 2>/dev/null || true)

            if (( ${#wandb_run_files[@]} > 0 )); then
                resume_wandb_run_id="$(basename "${wandb_run_files[0]}")"
                resume_wandb_run_id="${resume_wandb_run_id#run-}"
                resume_wandb_run_id="${resume_wandb_run_id%.wandb}"
            fi
        fi

        if [[ -n "${resume_wandb_run_id}" ]]; then
            echo "[info] resume: wandb run id ${resume_wandb_run_id} (${#wandb_run_files[@]} run dir(s) found)"
        else
            # resume="must" cannot succeed without a prior run, so wandb would
            # abort the whole job over metric logging. Training itself needs
            # nothing from wandb, and logs/ keeps the full metric stream.
            echo "[warn] resume: no wandb run found under ${OUTPUT_DIR}/wandb" >&2
            echo "[warn] resume: wandb is disabled for this run because" >&2
            echo "[warn] resume: wandb.init(resume=\"must\") needs an existing run." >&2
            echo "[warn] resume: metrics still go to the log file below." >&2
            echo "[info] resume: pass WANDB_RUN_ID=<id> to attach to a known run instead." >&2
            WANDB_ENABLED=0
        fi
    fi
elif [[ -d "${POLICY_PATH}" ]]; then
    for required in config.json model.safetensors; do
        if [[ ! -f "${POLICY_PATH}/${required}" ]]; then
            echo "[error] Policy checkpoint is incomplete: ${POLICY_PATH}" >&2
            echo "[error] Missing ${required}" >&2
            exit 5
        fi
    done
    echo "[info] policy checkpoint: local dir ${POLICY_PATH}"
elif [[ -n "${POLICY_PATH}" && "${HF_HUB_OFFLINE}" == "1" ]]; then
    echo "[error] POLICY_PATH=${POLICY_PATH} is a Hub id, but this run is offline" >&2
    echo "[error] and no local copy exists at:" >&2
    echo "[error]   ${POLICY_LOCAL_DIR}" >&2
    echo "[info] Download it on a networked machine and copy it there:" >&2
    echo "  hf download ${POLICY_REPO_ID} --local-dir weights/${POLICY_REPO_ID}" >&2
    echo "[info] Or allow downloads for this run with HF_HUB_OFFLINE=0." >&2
    exit 5
elif [[ -n "${POLICY_PATH}" ]]; then
    echo "[info] policy checkpoint: Hub id ${POLICY_PATH} (downloads enabled)"
fi

# Fail before the training process starts if the environment is incomplete.
WANDB_ENABLED="${WANDB_ENABLED}" python - <<'DEPS'
import importlib.metadata as md
import os

# train.py imports transformers only after the dataset is built, so a
# missing dependency would otherwise surface minutes into the run.
required = ["torch", "transformers", "accelerate", "draccus", "av", "datasets"]
if os.environ.get("WANDB_ENABLED") == "1":
    required.append("wandb")
missing = []
versions = {}
for name in required:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit(
        "[error] Missing packages in this environment: " + ", ".join(missing)
        + "\n[info] Build the supported environment with ./setup_vlab_env.sh, "
        "then re-run with CONDA_ENV=vlab."
    )

# datasets>=4 returns Column objects where _query_hf_dataset expects lists,
# which breaks torch.stack. setup_vlab_env.sh pins 3.6.0.
if int(versions["datasets"].split(".")[0]) >= 4:
    raise SystemExit(
        "[error] datasets==" + versions["datasets"] + " is incompatible with "
        "this code: torch.stack receives a Column, not tensors.\n"
        "[info] Use the supported environment: ./setup_vlab_env.sh, then "
        "re-run with CONDA_ENV=vlab."
    )
print("[info] deps: " + ", ".join(k + "=" + v for k, v in versions.items()))
DEPS

# Fail before the training process starts if CUDA or PyAV is broken.
EXPECTED_GPU_COUNT="${NUM_PROCESSES}" python - <<'PY'
import torch
import av

print(f"[info] PyTorch: {torch.__version__}; PyAV: {av.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
import os

expected = int(os.environ["EXPECTED_GPU_COUNT"])
actual = torch.cuda.device_count()
print(f"[info] visible GPUs: {actual} (need {expected})")
if actual < expected:
    raise RuntimeError(f"Requested {expected} processes but only {actual} GPUs are visible.")
for index in range(expected):
    major, minor = torch.cuda.get_device_capability(index)
    print(f"[info] GPU {index}: {torch.cuda.get_device_name(index)} (sm_{major}{minor})")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)

# Verify the GPU by running a kernel rather than by comparing against
# torch.cuda.get_arch_list(). A build without an exact sm_XX match can still be
# fine via forward-compatible PTX (e.g. sm_86 code runs on sm_89), so the list
# comparison used by the cluster scripts reports false failures here.
try:
    probe = torch.randn(256, 256, device="cuda")
    torch.matmul(probe, probe).sum().item()
    torch.cuda.synchronize()
except Exception as error:
    raise RuntimeError(
        f"GPU 0 ({name}, sm_{major}{minor}) cannot execute kernels with this "
        f"PyTorch build ({torch.__version__}): {error}"
    ) from error
print("[info] GPU kernel probe: OK")
PY

# --- Training arguments ---------------------------------------------------
train_args=(
    --policy.vlm_model_name="${VLM_MODEL_NAME}"
    --policy.push_to_hub=false
    --policy.optimizer_lr="${OPTIMIZER_LR}"
    --policy.train_expert_only="${TRAIN_EXPERT_ONLY}"
    --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER}"
    --dataset.repo_id="${DATASET_REPO_IDS}"
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
    --dataset.val_episodes_per_dataset="${VAL_EPISODES_PER_DATASET}"
    --dataset.val_split_seed="${VAL_SPLIT_SEED}"
    --val_max_batches="${VAL_MAX_BATCHES}"
    --val_action_batches="${VAL_ACTION_BATCHES}"
    --policy.use_amp="${USE_AMP}"
    --wandb.project="${WANDB_PROJECT}"
    --wandb.mode="${WANDB_MODE}"
    --trackio.enable=false
)

if [[ "${RESUME}" == "1" ]]; then
    # --config_path reloads the whole train config from the checkpoint
    # (parser.py:220 -> configs/train.py:95), and the arguments assembled above
    # are applied on top of it as CLI overrides. --policy.path is deliberately
    # absent: configs/train.py:85 rejects it together with --resume=true,
    # because the two answer the same question differently.
    #
    # Two of the overrides above are inert here. The optimizer's param_groups,
    # lr included, are restored from the checkpoint (optimizers.py:223), and so
    # is the scheduler state, so OPTIMIZER_LR and the warmup/decay settings
    # cannot be changed by a resume -- leave them at the original run's values
    # to keep the printed summary honest.
    train_args+=(
        --config_path="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
        --resume=true
    )
    # Supplying the id skips get_wandb_run_id_from_filesystem entirely
    # (wandb_utils.py:82), so the resume no longer depends on the latest-run
    # symlink surviving the crash that made the resume necessary.
    if [[ "${WANDB_ENABLED}" == "1" && -n "${resume_wandb_run_id}" ]]; then
        train_args+=(--wandb.run_id="${resume_wandb_run_id}")
    fi
    policy_init="resume:${RESUME_CHECKPOINT}@${resume_step}"
elif [[ -n "${POLICY_PATH}" ]]; then
    # Fine-tune a complete SmolVLA checkpoint. load_vlm_weights stays false:
    # the backbone arrives with the checkpoint, so re-loading the base SmolVLM
    # shards would overwrite the fine-tuned ones.
    train_args+=(
        --policy.path="${POLICY_PATH}"
        --policy.load_vlm_weights=false
    )
    policy_init="pretrained:${POLICY_PATH}"
else
    # Randomly initialised action expert. Only viable with far more data than
    # the TeleAvatar sets hold; run03 took this path and produced a policy that
    # only jittered in place.
    echo "[warn] POLICY_PATH is empty: the action expert will be RANDOMLY" >&2
    echo "[warn] INITIALISED and must be learned from scratch. This is what" >&2
    echo "[warn] run03 did. Set POLICY_PATH to a pretrained checkpoint to" >&2
    echo "[warn] fine-tune instead." >&2
    train_args+=(
        --policy.type=smolvla2
        --policy.load_vlm_weights=true
    )
    policy_init="scratch:smolvla2"
fi

if [[ -n "${DATASET_SAMPLING_WEIGHTS}" ]]; then
    train_args+=(--dataset.sampling_weights="${DATASET_SAMPLING_WEIGHTS}")
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
    train_args+=(--wandb.enable=true)
    if [[ "${WANDB_DISABLE_ARTIFACT}" == "1" ]]; then
        train_args+=(--wandb.disable_artifact=true)
    else
        train_args+=(--wandb.disable_artifact=false)
    fi
else
    train_args+=(--wandb.enable=false)
fi

global_batch=$(( NUM_PROCESSES * BATCH_SIZE ))
echo "[info] experiment=${EXP_NAME}"
echo "[info] datasets=${num_datasets} (${DATASET_REPO_IDS})"
echo "[info] totals: ${total_episodes} episodes, ${total_frames} frames"
if (( total_frames > 0 )); then
    # Sanity check on the step budget: too few epochs underfits, too many on a
    # small set overfits. run03 ran ~68 epochs over 4695 frames.
    echo "[info] step budget: $(( total_frames / global_batch )) steps/epoch," \
         "${NUM_TRAIN_STEPS} steps = $(( NUM_TRAIN_STEPS * global_batch / total_frames )) epochs"
fi
echo "[info] adapter: 72-dim -> state 16 (arms14 + measured gripper positions2) / action 16; 3 cameras cropped to left eye and downscaled to the deploy feed (head 960x960, wrists 400x640)"
echo "[info] policy_init=${policy_init} optimizer_lr=${OPTIMIZER_LR} train_expert_only=${TRAIN_EXPERT_ONLY} freeze_vision_encoder=${FREEZE_VISION_ENCODER}"
echo "[info] GPUs=${NUM_PROCESSES} batch_per_gpu=${BATCH_SIZE} global_batch=${global_batch}"
echo "[info] workers_per_process=${NUM_WORKERS} steps=${NUM_TRAIN_STEPS} output=${OUTPUT_DIR}"
echo "[info] scheduler_warmup_steps=${SCHEDULER_WARMUP_STEPS} scheduler_decay_steps=${SCHEDULER_DECAY_STEPS} scheduler_decay_lr=${SCHEDULER_DECAY_LR}"
echo "[info] wandb_enabled=${WANDB_ENABLED} wandb_mode=${WANDB_MODE} disable_artifact=${WANDB_DISABLE_ARTIFACT} data_dir=${WANDB_DATA_DIR}"
if (( VAL_EPISODES_PER_DATASET > 0 )); then
    echo "[info] validation: ${VAL_EPISODES_PER_DATASET} held-out episode(s)/dataset, seed=${VAL_SPLIT_SEED}," \
         "${VAL_MAX_BATCHES} loss batches + ${VAL_ACTION_BATCHES} rollout batches every ${SAVE_FREQ} steps"
else
    echo "[info] validation: disabled (training on every episode)"
fi

log_file="logs/${EXP_NAME}-$(date +%Y%m%d-%H%M%S).log"
echo "[info] log=${log_file}"

if (( NUM_PROCESSES > 1 )); then
    accelerate_config="${PROJECT_ROOT}/accelerate_configs/multi_gpu.yaml"
else
    accelerate_config="${PROJECT_ROOT}/accelerate_configs/single_gpu.yaml"
fi

# --num_processes overrides the num_processes value inside the YAML.
accelerate launch \
    --config_file "${accelerate_config}" \
    --num_processes "${NUM_PROCESSES}" \
    "${PROJECT_ROOT}/src/lerobot/scripts/train.py" \
    "${train_args[@]}" 2>&1 | tee "${log_file}"

echo "[info] Training completed: ${OUTPUT_DIR}"
