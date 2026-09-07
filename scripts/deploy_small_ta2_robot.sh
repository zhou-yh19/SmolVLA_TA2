#!/usr/bin/env bash
# Deploy a trained SmolVLA2 checkpoint onto the TeleAvatar V2 robot.
#
# This is a thin, opinionated wrapper around teleavatar_v2/scripts/run_smolvla.py.
# That script and smolvla_deploy/ hold the real logic; this one only fixes the
# paths, resolves a checkpoint step, and enforces the staged bring-up order.
#
# Run it on the ROBOT host, not the training server. Stages:
#   ./scripts/deploy_small_ta2_robot.sh check     # env + checkpoint, no ROS
#   ./scripts/deploy_small_ta2_robot.sh cameras   # decode RTP, save six crops
#   ./scripts/deploy_small_ta2_robot.sh observe   # print the exact observation
#   ./scripts/deploy_small_ta2_robot.sh zero      # move both arms to zero
#   ./scripts/deploy_small_ta2_robot.sh dry       # one inference chunk, no commands
#   ./scripts/deploy_small_ta2_robot.sh run       # PUBLISHES COMMANDS
#
# Every value can be overridden from the environment:
#   CKPT_STEP=002000 TASK="pick up the block" ./scripts/deploy_small_ta2_robot.sh dry
set -Eeuo pipefail

# --- Paths ----------------------------------------------------------------
# On the robot host, not the training server. Override both if they differ.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPLOY_ROOT="${DEPLOY_ROOT:-${PROJECT_ROOT}/teleavatar_v2}"
CONDA_ENV="${CONDA_ENV:-teleavatar-smolvla}"

EXP_NAME="${EXP_NAME:-smolvla_ta2_multi_run06}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${EXP_NAME}}"

# Which checkpoint to deploy. SAVE_FREQ was 2000, so run03 has 002000, 004000,
# ... 010000 plus "last". Training loss flattened by ~2000 steps and the run
# went to 10000 with no validation set, so the late checkpoints are the ones
# most likely to have overfit 4695 frames. Start at 002000 and walk forward.
CKPT_STEP="${CKPT_STEP:-last}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/${CKPT_STEP}/pretrained_model}"

# The VLM config/processor must already be on this host: deployment loads the
# policy weights from the checkpoint, but still needs the tokenizer/processor.
VLM_REPO_ID="${VLM_REPO_ID:-HuggingFaceTB/SmolVLM2-500M-Video-Instruct}"
VLM_LOCAL_DIR="${VLM_LOCAL_DIR:-${PROJECT_ROOT}/weights/${VLM_REPO_ID}}"

# --- Policy / control -----------------------------------------------------
# The language instruction. It MUST match the phrasing used in the training
# episodes: the policy conditions on it and an unseen phrasing degrades it.
TASK="${TASK:-a base layer is already in place, pick left-side block with the left arm and right-side block with the right arm, place block on the base layer.}"
DEVICE="${DEVICE:-cuda}"

# Actions are executed open-loop at this rate. The dataset is 30fps; 20Hz
# leaves headroom for inference between chunks.
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-20}"
# Of the 50 predicted actions, execute this many, then replan from a fresh
# observation. Smaller = more reactive and more inference; larger = smoother
# but longer blind. 16 at 20Hz replans every 0.8s.
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"
# Per-command joint-step clamp, in radians. This is the main safety limit: it
# caps how far one command can jump from the previous one, so a bad prediction
# ramps instead of snapping. 0.05 rad at 20Hz is ~1 rad/s. Raise only after
# the policy has proven itself; never start above 0.10.
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.05}"
# Stop after N chunks; 0 is unlimited. A small number is useful for first runs.
MAX_CHUNKS="${MAX_CHUNKS:-0}"

# --- ROS2 -----------------------------------------------------------------
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-19}"
export ROS_DISTRO="${ROS_DISTRO:-humble}"
RTP_PORT="${RTP_PORT:-8890}"
CAMERA_DUMP_DIR="${CAMERA_DUMP_DIR:-${OUTPUT_DIR}/teleavatar_cameras}"

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit 1
fi

# --- Environment ----------------------------------------------------------
if [[ -n "${CONDA_ENV}" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    for base in "${CONDA_PREFIX_1:-}" "${HOME}/miniconda3" "${HOME}/anaconda3" /opt/conda; do
        if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "${base}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV}"
            break
        fi
    done
    if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
        echo "[warn] could not activate conda env '${CONDA_ENV}'; using current interpreter" >&2
    fi
fi

if [[ ! -d "${DEPLOY_ROOT}" ]]; then
    echo "[error] deployment directory not found: ${DEPLOY_ROOT}" >&2
    exit 3
fi

echo "[info] stage=${STAGE} exp=${EXP_NAME} step=${CKPT_STEP}"
echo "[info] checkpoint=${CHECKPOINT}"
echo "[info] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} rtp_port=${RTP_PORT}"

# --- Checkpoint preflight -------------------------------------------------
# Runs for every stage that loads the policy. Failing here costs seconds;
# failing after the arms are enabled does not.
check_checkpoint() {
    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "[error] checkpoint not found: ${CHECKPOINT}" >&2
        if [[ -d "${OUTPUT_DIR}/checkpoints" ]]; then
            echo "[info] available steps in ${OUTPUT_DIR}/checkpoints:" >&2
            ls -1 "${OUTPUT_DIR}/checkpoints" >&2
        else
            echo "[info] no checkpoints directory at ${OUTPUT_DIR}/checkpoints" >&2
            echo "[info] copy the trained run over from the training server first." >&2
        fi
        exit 4
    fi
    for required in config.json model.safetensors; do
        if [[ ! -f "${CHECKPOINT}/${required}" ]]; then
            echo "[error] ${CHECKPOINT} is missing ${required}" >&2
            echo "[info] point --checkpoint at the pretrained_model/ subdirectory." >&2
            exit 4
        fi
    done

    # The runtime re-validates all of this at startup and refuses to command the
    # robot if it disagrees. Reading it here just makes a mismatch legible
    # before ROS2 and the arms are involved. It uses the same contract helpers
    # as the runtime so the two can't drift apart.
    CHECKPOINT="${CHECKPOINT}" DEPLOY_ROOT="${DEPLOY_ROOT}" python - <<'PY'
import json, os, pathlib, sys

sys.path.insert(0, os.environ["DEPLOY_ROOT"])
from smolvla_deploy.contracts import ACTION_DIM, STATE_DIM, resolve_checkpoint_width, validate_state_layout

path = pathlib.Path(os.environ["CHECKPOINT"]) / "config.json"
config = json.loads(path.read_text())
print(f"[info] policy type: {config.get('type')}")

problems = []
try:
    validate_state_layout(config.get("teleavatar_state_layout"))
except ValueError as error:
    problems.append(str(error))
features = config.get("input_features") or {}
state = (features.get("observation.state") or {}).get("shape")
action = ((config.get("output_features") or {}).get("action") or {}).get("shape")
print(f"[info] state={state} action={action}")

# A multi-dataset run pads both to the policy's max_*_dim; the TeleAvatar values
# still occupy the leading dimensions, so such a checkpoint is deployable.
widths = {}
for name, shape, real_dim, max_key in (
    ("observation.state", state, STATE_DIM, "max_state_dim"),
    ("action", action, ACTION_DIM, "max_action_dim"),
):
    if not shape:
        continue
    try:
        widths[name] = resolve_checkpoint_width(name, shape, real_dim, config.get(max_key))
    except ValueError as error:
        problems.append(str(error))
    else:
        if widths[name] != real_dim:
            print(f"[info] {name} padded to {widths[name]}; first {real_dim} dims are TeleAvatar")

cameras = sorted(k for k in features if k.startswith("observation.images."))
expected_cameras = [
    "observation.images.image",
    "observation.images.image2",
    "observation.images.image3",
]
print(f"[info] cameras: {cameras}")
if cameras and cameras != expected_cameras:
    problems.append(f"cameras {cameras}, expected {expected_cameras}")

for key, want in (
    ("n_obs_steps", 1),
    ("predict_relative_actions", False),
    ("adapt_to_pi_aloha", False),
):
    if key in config and config[key] != want:
        problems.append(f"{key}={config[key]}, expected {want}")

chunk = config.get("chunk_size")
if chunk:
    print(f"[info] chunk_size={chunk}")

if problems:
    raise SystemExit("[error] checkpoint/runtime contract mismatch:\n  - " + "\n  - ".join(problems))
print("[info] checkpoint contract: OK")
PY
}

resolve_vlm_flag() {
    VLM_FLAG=()
    if [[ -f "${VLM_LOCAL_DIR}/config.json" ]]; then
        VLM_FLAG=(--vlm-model-path "${VLM_LOCAL_DIR}")
        echo "[info] VLM processor: local ${VLM_LOCAL_DIR}"
    else
        echo "[warn] no local VLM copy at ${VLM_LOCAL_DIR}" >&2
        echo "[warn] the runtime will need ${VLM_REPO_ID} in the HF cache or network access." >&2
    fi
}

cd "${DEPLOY_ROOT}"

case "${STAGE}" in
check)
    # Dependencies first: the runtime imports torch and the repo lazily, so a
    # broken env would otherwise surface only after ROS2 is up.
    python - <<'PY'
import importlib.metadata as md

missing, versions = [], {}
for name in ["torch", "transformers", "safetensors", "numpy", "PyYAML"]:
    try:
        versions[name] = md.version(name)
    except md.PackageNotFoundError:
        missing.append(name)
if missing:
    raise SystemExit("[error] missing packages: " + ", ".join(missing)
                     + "\n[info] conda env create -n teleavatar-smolvla -f environment.yml")
print("[info] deps: " + ", ".join(f"{k}={v}" for k, v in versions.items()))

import torch
print(f"[info] PyTorch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is not available on this host.")
print(f"[info] GPU: {torch.cuda.get_device_name(0)}")
# Probe with a real kernel: comparing against get_arch_list() reports false
# failures when a build runs via forward-compatible PTX.
probe = torch.randn(256, 256, device="cuda")
torch.matmul(probe, probe).sum().item()
torch.cuda.synchronize()
print("[info] GPU kernel probe: OK")
PY
    check_checkpoint
    echo "[info] offline contract tests:"
    python -m unittest discover -s tests
    echo "[info] check passed. Next: '${0##*/} cameras' with the RTP stream running."
    ;;

cameras)
    # Saves all six crops. Open them and confirm the three left-eye views match
    # what training saw: same mounting, similar lighting, nothing occluded.
    echo "[info] writing crops to ${CAMERA_DUMP_DIR}"
    python scripts/test_video.py \
        --port "${RTP_PORT}" \
        --split-output-dir "${CAMERA_DUMP_DIR}" \
        --duration-s "${CAMERA_DURATION_S:-5}"
    echo "[info] inspect the crops before continuing:"
    echo "       ls ${CAMERA_DUMP_DIR}"
    ;;

observe)
    # Prints the exact 16-D state and image keys handed to the policy. Verify
    # joint values look sane and left/right are not swapped.
    python scripts/inspect_observation.py \
        --arm-config "${DEPLOY_ROOT}/arm_config.yml" \
        --rtp-port "${RTP_PORT}"
    ;;

zero)
    echo "[warn] this MOVES BOTH ARMS to zero. Clear the workspace." >&2
    read -r -p "Type 'zero' to continue: " confirm
    [[ "${confirm}" == "zero" ]] || { echo "[info] aborted"; exit 1; }
    python scripts/zero.py
    ;;

dry|run)
    check_checkpoint
    resolve_vlm_flag

    run_args=(
        --checkpoint "${CHECKPOINT}"
        --smolvla-repo "${PROJECT_ROOT}"
        "${VLM_FLAG[@]}"
        --device "${DEVICE}"
        --task "${TASK}"
        --control-frequency "${CONTROL_FREQUENCY}"
        --execution-horizon "${EXECUTION_HORIZON}"
        --max-joint-step-rad "${MAX_JOINT_STEP_RAD}"
        --rtp-port "${RTP_PORT}"
        --arm-config "${DEPLOY_ROOT}/arm_config.yml"
    )
    [[ "${MAX_CHUNKS}" != "0" ]] && run_args+=(--max-chunks "${MAX_CHUNKS}")

    echo "[info] task: ${TASK}"

    if [[ "${STAGE}" == "dry" ]]; then
        # No --execute: infers one chunk, logs shape and triggers, exits without
        # publishing. Check that both trigger values sit inside [0,1] and are not
        # pinned at a constant before enabling execution.
        echo "[info] DRY RUN: one chunk, no commands published"
        exec python scripts/run_smolvla.py "${run_args[@]}"
    fi

    echo "[warn] ============================================" >&2
    echo "[warn]  THIS WILL COMMAND THE ROBOT" >&2
    echo "[warn]  freq=${CONTROL_FREQUENCY}Hz horizon=${EXECUTION_HORIZON} step_limit=${MAX_JOINT_STEP_RAD} rad" >&2
    echo "[warn]  Keep the e-stop in hand. Ctrl+C stops the loop." >&2
    echo "[warn]  Nothing else may publish to the arm command topics." >&2
    echo "[warn] ============================================" >&2
    read -r -p "Type 'execute' to continue: " confirm
    [[ "${confirm}" == "execute" ]] || { echo "[info] aborted"; exit 1; }

    exec python scripts/run_smolvla.py "${run_args[@]}" --execute
    ;;

*)
    echo "[error] unknown stage '${STAGE}'" >&2
    echo "[info] stages: check | cameras | observe | zero | dry | run" >&2
    exit 1
    ;;
esac
