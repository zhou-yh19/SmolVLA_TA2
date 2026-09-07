#!/usr/bin/env bash
# Verify the TeleAvatar V2 adapter output before committing to a long run.
#
# README asks for this check: state [B,16], action [B,horizon,16], and the three
# canonical camera keys. Runs on CPU, decodes a couple of frames only.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/arapat/disk0/SmolVLA_TA2}"
CONDA_ENV="${CONDA_ENV:-vlab}"
DATASET_ROOT="${DATASET_ROOT:-/DATA/disk0/arapat/SmolVLA_TA2/datasets}"
DATASET_REPO_ID="${DATASET_REPO_ID:-small_lerobot_30fps}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

DATASET_ROOT="${DATASET_ROOT}" DATASET_REPO_ID="${DATASET_REPO_ID}" \
CHUNK_SIZE="${CHUNK_SIZE}" BATCH_SIZE="${BATCH_SIZE}" python - <<'CHECK'
import os
from pathlib import Path

import torch

from lerobot.datasets.adapters import get_dataset_adapter
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

root = Path(os.environ["DATASET_ROOT"]) / os.environ["DATASET_REPO_ID"]
repo_id = os.environ["DATASET_REPO_ID"]
chunk = int(os.environ["CHUNK_SIZE"])
batch_size = int(os.environ["BATCH_SIZE"])

meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=True)
print(f"robot_type: {meta.robot_type}")
adapter = get_dataset_adapter(meta.robot_type, dataset_root=meta.root)
if adapter is None:
    raise SystemExit(
        f"[error] No adapter for robot_type={meta.robot_type!r}. Training would "
        "receive raw 72-dim state/action."
    )
print(f"adapter: {type(adapter).__name__}")

fps = meta.info["fps"]
delta_timestamps = {"action": [i / fps for i in range(chunk)]}
dataset = LeRobotDataset(
    repo_id,
    root=root,
    local_files_only=True,
    delta_timestamps=delta_timestamps,
    video_backend="pyav",
    feature_adapter=adapter,
)
print(f"frames: {dataset.num_frames}  episodes: {dataset.num_episodes}  fps: {fps}")

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
batch = next(iter(loader))

expected = {
    "observation.state": (batch_size, 16),
    "action": (batch_size, chunk, 16),
}
failures = []
for key, want in expected.items():
    if key not in batch:
        failures.append(f"{key}: MISSING")
        continue
    got = tuple(batch[key].shape)
    status = "OK" if got == want else "MISMATCH"
    if got != want:
        failures.append(f"{key}: got {got}, want {want}")
    print(f"{key}: {got} [{status}]")

for key in ("observation.images.image", "observation.images.image2", "observation.images.image3"):
    if key not in batch:
        failures.append(f"{key}: MISSING")
        print(f"{key}: MISSING")
    else:
        print(f"{key}: {tuple(batch[key].shape)} [OK]")

stale = [k for k in batch if any(s in k for s in ("head_camera", "left_color", "right_color"))]
if stale:
    failures.append(f"unmapped source camera keys still present: {stale}")
print(f"unmapped source camera keys: {stale or 'none'}")

pad = batch.get("action_is_pad")
print(f"action_is_pad: {tuple(pad.shape) if pad is not None else 'MISSING'}")
if pad is None:
    failures.append("action_is_pad missing: episode-boundary masking would be inactive")

action = batch["action"]
print(f"action[..., 7] gripper range:  [{action[..., 7].min():.3f}, {action[..., 7].max():.3f}]")
print(f"action[..., 15] gripper range: [{action[..., 15].min():.3f}, {action[..., 15].max():.3f}]")
print(f"task: {str(batch['task'][0])[:60]}...")

if failures:
    print("\n[error] Checks failed:")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)
print("\n[info] All batch checks passed.")
CHECK
