"""Pure data contracts shared by the policy and robot sides.

This module deliberately has no ROS2, GStreamer, or PyTorch dependency so the
most safety-critical shape and semantic checks can be tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

STATE_DIM = 14
ACTION_DIM = 16

OBS_STATE = "observation.state"
OBS_IMAGE = "observation.images.image"
OBS_IMAGE_2 = "observation.images.image2"
OBS_IMAGE_3 = "observation.images.image3"

# The SmolVLA training adapter crops the left eye from all three stereo views.
SMOLVLA_CAMERA_MAPPING = {
    OBS_IMAGE: "head_left_eye",
    OBS_IMAGE_2: "left_wrist_left_eye",
    OBS_IMAGE_3: "right_wrist_left_eye",
}


@dataclass(frozen=True)
class TriggerAction:
    """TeleAvatar command values after validating the SmolVLA action contract."""

    left_arm: np.ndarray
    left_trigger: float
    right_arm: np.ndarray
    right_trigger: float


def build_state(left_positions: np.ndarray, right_positions: np.ndarray) -> np.ndarray:
    """Build the exact 14-D state used during TeleAvatar SmolVLA training."""
    left = np.asarray(left_positions, dtype=np.float32)
    right = np.asarray(right_positions, dtype=np.float32)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError(f"Expected two 7-D arm states, got {left.shape} and {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Joint state contains NaN or infinity")
    return np.concatenate((left, right), dtype=np.float32)


def map_smolvla_images(split_images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map RTP split names to the three canonical SmolVLA image keys."""
    missing = [view for view in SMOLVLA_CAMERA_MAPPING.values() if view not in split_images]
    if missing:
        raise KeyError(f"Missing RTP camera views: {missing}")

    result: dict[str, np.ndarray] = {}
    for model_key, view in SMOLVLA_CAMERA_MAPPING.items():
        image = np.asarray(split_images[view])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera {view!r} must be HWC RGB, got {image.shape}")
        result[model_key] = image.copy()
    return result


def split_trigger_action(action: np.ndarray) -> TriggerAction:
    """Validate one 16-D SmolVLA action and expose direct trigger commands.

    Dimensions 7 and 15 are already TeleAvatar trigger values. They are only
    clipped to the hardware range; no effort conversion is performed.
    """
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape ({ACTION_DIM},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Action contains NaN or infinity")
    return TriggerAction(
        left_arm=value[0:7].copy(),
        left_trigger=float(np.clip(value[7], 0.0, 1.0)),
        right_arm=value[8:15].copy(),
        right_trigger=float(np.clip(value[15], 0.0, 1.0)),
    )


def select_execution_actions(action_chunk: np.ndarray, execution_horizon: int) -> np.ndarray:
    """Validate a full policy chunk and select the open-loop prefix to execute."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action chunk [T,{ACTION_DIM}], got {chunk.shape}")
    if chunk.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if not np.isfinite(chunk).all():
        raise ValueError("Action chunk contains NaN or infinity")
    if execution_horizon <= 0:
        raise ValueError("execution_horizon must be positive")
    return chunk[: min(execution_horizon, chunk.shape[0])].copy()
