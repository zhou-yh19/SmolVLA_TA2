"""Pure data contracts shared by the policy and robot sides.

This module deliberately has no ROS2, GStreamer, or PyTorch dependency so the
most safety-critical shape and semantic checks can be tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

# The real TeleAvatar widths: seven joints per arm, plus one gripper trigger per
# arm on the action side. A checkpoint may store them padded; see
# `resolve_checkpoint_width`.
STATE_DIM = 16
STATE_LAYOUT = "arms14_gripper_positions2_v1"
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


def resolve_checkpoint_width(name: str, shape, real_dim: int, max_dim: int | None) -> int:
    """Return the width at which a checkpoint stores a TeleAvatar feature.

    A single-dataset run stores the feature at its real width. A multi-dataset
    run pads every dataset to the policy's `max_state_dim` / `max_action_dim` so
    the batch is rectangular, so both the config and the normalization buffers
    are that wide instead: the TeleAvatar values occupy the leading `real_dim`
    slots and the rest is zero-mean/unit-std padding. Both are deployable, but
    the runtime has to feed and slice tensors at whichever width it finds.
    """
    width = list(shape)
    if width == [real_dim]:
        return real_dim
    if max_dim is not None and width == [max_dim] and max_dim > real_dim:
        return max_dim
    expected = f"[{real_dim}]"
    if max_dim is not None and max_dim > real_dim:
        expected += f" or [{max_dim}] (padded)"
    raise ValueError(f"Checkpoint {name} shape is {width}, expected {expected}")


def pad_to_width(vector: np.ndarray, width: int) -> np.ndarray:
    """Zero-pad a TeleAvatar vector to the width a padded checkpoint expects.

    Training padded with zeros before computing statistics, so the padded slots
    normalize to exactly zero and contribute nothing.
    """
    value = np.asarray(vector, dtype=np.float32)
    if value.shape[-1] > width:
        raise ValueError(f"Cannot pad {value.shape[-1]} values down to width {width}")
    if value.shape[-1] == width:
        return value
    padding = np.zeros((*value.shape[:-1], width - value.shape[-1]), dtype=np.float32)
    return np.concatenate((value, padding), axis=-1, dtype=np.float32)


def validate_state_layout(layout: str | None) -> None:
    if layout != STATE_LAYOUT:
        raise ValueError(
            f"Checkpoint state layout {layout!r} is incompatible with {STATE_LAYOUT!r}. "
            "Use a checkpoint trained with measured gripper positions; older padded "
            "32-D checkpoints cannot be identified by tensor width alone."
        )


def gripper_position(positions) -> float:
    """Read the single measured position from a TA2 gripper JointState."""
    values = np.asarray(positions, dtype=np.float32)
    if values.shape != (1,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite measured gripper position")
    return float(values[0])


def build_state(
    left_positions: np.ndarray, right_positions: np.ndarray,
    left_gripper_position: float, right_gripper_position: float,
) -> np.ndarray:
    """Build [left arm 7, right arm 7, left gripper position, right gripper position]."""
    left = np.asarray(left_positions, dtype=np.float32)
    right = np.asarray(right_positions, dtype=np.float32)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError(f"Expected two 7-D arm states, got {left.shape} and {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Joint state contains NaN or infinity")
    grippers = np.asarray([left_gripper_position, right_gripper_position], dtype=np.float32)
    if grippers.shape != (2,) or not np.isfinite(grippers).all():
        raise ValueError("Expected two finite scalar gripper positions")
    return np.concatenate((left, right, grippers), dtype=np.float32)


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
