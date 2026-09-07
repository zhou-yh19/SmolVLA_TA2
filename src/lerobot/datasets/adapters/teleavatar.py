#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TeleAvatar V2 dataset adaptation for SmolVLA.

The source TA2 dataset stores 72-dimensional state and action vectors. SmolVLA
uses 14 arm joint positions as state and predicts 16 absolute controls: seven
joint positions and one gripper trigger for each arm. All three stereo cameras
are cropped to their left eye, downscaled to the resolution the robot streams
at deployment, and mapped to stable SmolVLA camera keys.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ACTION = "action"
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.images.image"
OBS_IMAGE_2 = "observation.images.image2"
OBS_IMAGE_3 = "observation.images.image3"
TELEAVATAR_V2_STATS_PATH = Path("meta/teleavatar_v2_stats.json")

TELEAVATAR_JOINT_POSITION_NAMES = (
    "left_joint1_position",
    "left_joint2_position",
    "left_joint3_position",
    "left_joint4_position",
    "left_joint5_position",
    "left_joint6_position",
    "left_joint7_position",
    "left_gripper_position",
    "right_joint1_position",
    "right_joint2_position",
    "right_joint3_position",
    "right_joint4_position",
    "right_joint5_position",
    "right_joint6_position",
    "right_joint7_position",
    "right_gripper_position",
)

TELEAVATAR_STATE_LAYOUT = "arms14_gripper_positions2_v1"
TELEAVATAR_STATE_INDICES = (*range(7), *range(8, 15), 7, 15)
TELEAVATAR_ACTION_INDICES = (*range(7), 39, *range(8, 15), 47)

TELEAVATAR_STATE_NAMES = tuple(
    TELEAVATAR_JOINT_POSITION_NAMES[index] for index in TELEAVATAR_STATE_INDICES
)
TELEAVATAR_ACTION_SOURCE_NAMES = (
    *TELEAVATAR_JOINT_POSITION_NAMES[:7],
    "left_gripper_effort",
    *TELEAVATAR_JOINT_POSITION_NAMES[8:15],
    "right_gripper_effort",
)
TELEAVATAR_ACTION_NAMES = (
    *TELEAVATAR_JOINT_POSITION_NAMES[:7],
    "left_gripper_trigger",
    *TELEAVATAR_JOINT_POSITION_NAMES[8:15],
    "right_gripper_trigger",
)

TELEAVATAR_RAW_FEATURE_NAMES = (
    *TELEAVATAR_JOINT_POSITION_NAMES,
    *(
        name.replace("_position", "_velocity")
        for name in TELEAVATAR_JOINT_POSITION_NAMES
    ),
    *(name.replace("_position", "_effort") for name in TELEAVATAR_JOINT_POSITION_NAMES),
    "left_ee_position_x",
    "left_ee_position_y",
    "left_ee_position_z",
    "left_ee_orientation_x",
    "left_ee_orientation_y",
    "left_ee_orientation_z",
    "left_ee_orientation_w",
    "right_ee_position_x",
    "right_ee_position_y",
    "right_ee_position_z",
    "right_ee_orientation_x",
    "right_ee_orientation_y",
    "right_ee_orientation_z",
    "right_ee_orientation_w",
    "chassis_motor1_position",
    "chassis_motor2_position",
    "chassis_motor3_position",
    "chassis_motor1_velocity",
    "chassis_motor2_velocity",
    "chassis_motor3_velocity",
    "chassis_motor1_effort",
    "chassis_motor2_effort",
    "chassis_motor3_effort",
    "kinco_velocity",
)

TELEAVATAR_IMAGE_KEY_MAPPING = {
    "observation.images.head_camera": OBS_IMAGE,
    "observation.images.left_color": OBS_IMAGE_2,
    "observation.images.right_color": OBS_IMAGE_3,
}

# Per-eye (height, width) the deployed robot delivers. The dataset stores the
# full sensor resolution (head 1920x3840, wrists 800x2560 side-by-side stereo),
# but at run time the robot composites all six eyes into one 1280x2720 H265
# RTP stream and `teleavatar_v2/smolvla_deploy/rtp_video_interface.py` crops
# each eye back out of it, so the policy sees the head at 960x960 and each
# wrist at 400x640: exactly half the dataset resolution. Training frames are
# brought to the same size here so the policy's own 512x512 resize sees the
# same source resolution in both places. `None` keeps the full-resolution crop.
TELEAVATAR_DEPLOY_IMAGE_HW: dict[str, tuple[int, int]] = {
    OBS_IMAGE: (960, 960),
    OBS_IMAGE_2: (400, 640),
    OBS_IMAGE_3: (400, 640),
}


def gripper_effort_to_trigger(effort: np.ndarray | torch.Tensor):
    """Convert the TA2 gripper effort in Nm to the platform trigger space."""
    if isinstance(effort, torch.Tensor):
        return torch.where(
            effort > 0,
            0.10 * (1.0 - effort / 2.0),
            0.10 - effort * 0.90 / 1.6,
        )
    effort = np.asarray(effort)
    return np.where(effort > 0, 0.10 * (1.0 - effort / 2.0), 0.10 - effort * 0.90 / 1.6)


def adapt_teleavatar_state(state: np.ndarray | torch.Tensor):
    """Keep the 14 arm joints, then append measured left/right gripper positions."""
    return state[..., list(TELEAVATAR_STATE_INDICES)]


def adapt_teleavatar_action(action: np.ndarray | torch.Tensor):
    """Select absolute arm positions and convert both gripper efforts to triggers."""
    selected = (
        action[..., list(TELEAVATAR_ACTION_INDICES)].clone()
        if isinstance(action, torch.Tensor)
        else np.array(action[..., list(TELEAVATAR_ACTION_INDICES)], copy=True)
    )
    selected[..., 7] = gripper_effort_to_trigger(selected[..., 7])
    selected[..., 15] = gripper_effort_to_trigger(selected[..., 15])
    return selected


def crop_left_stereo_eye(image: np.ndarray | torch.Tensor):
    """Crop the left half of a stereo frame, preserving already-cropped images.

    LeRobot yields channel-first tensors, while deployment/debug helpers may
    provide channel-last arrays. Leading batch/time dimensions are supported.
    """
    if image.ndim < 3:
        raise ValueError(
            f"Expected an image with at least 3 dimensions, got {tuple(image.shape)}."
        )

    if image.shape[-3] in (1, 3, 4):  # (..., C, H, W)
        height, width = image.shape[-2:]
        if width >= 2 * height:
            return image[..., : width // 2]
        return image

    if image.shape[-1] in (1, 3, 4):  # (..., H, W, C)
        height, width = image.shape[-3:-1]
        if width >= 2 * height:
            return image[..., : width // 2, :]
        return image

    raise ValueError(
        f"Could not identify the channel dimension for image shape {tuple(image.shape)}."
    )


def _image_layout(image: np.ndarray | torch.Tensor) -> str:
    """Return "chw" or "hwc" for an image with optional leading batch dims."""
    if image.ndim < 3:
        raise ValueError(
            f"Expected an image with at least 3 dimensions, got {tuple(image.shape)}."
        )
    if image.shape[-3] in (1, 3, 4):
        return "chw"
    if image.shape[-1] in (1, 3, 4):
        return "hwc"
    raise ValueError(
        f"Could not identify the channel dimension for image shape {tuple(image.shape)}."
    )


def resize_image(image: np.ndarray | torch.Tensor, height: int, width: int):
    """Resize a single-eye frame to (height, width), matching the deployment feed.

    Accepts the channel-first float tensors LeRobot decodes from video and the
    channel-last uint8 arrays used by deployment and debugging helpers, with any
    number of leading batch/time dimensions. Returns the same layout and dtype.
    An integer reduction (the deployment case is exactly 2x) is a box filter,
    i.e. each output pixel is the mean of its source block, so no source pixel
    is dropped; other ratios fall back to antialiased bilinear resampling.
    Already-matching frames are returned untouched.
    """
    layout = _image_layout(image)
    current_hw = (
        tuple(image.shape[-2:]) if layout == "chw" else tuple(image.shape[-3:-1])
    )
    if current_hw == (height, width):
        return image

    is_numpy = isinstance(image, np.ndarray)
    tensor = torch.from_numpy(np.ascontiguousarray(image)) if is_numpy else image
    if layout == "hwc":
        tensor = tensor.movedim(-1, -3)

    lead_shape = tensor.shape[:-3]
    channels = tensor.shape[-3]
    flat = tensor.reshape(-1, channels, *tensor.shape[-2:])
    source_dtype = flat.dtype
    if not flat.is_floating_point():
        flat = flat.to(torch.float32)
    source_height, source_width = flat.shape[-2:]
    integer_reduction = (
        source_height >= height
        and source_width >= width
        and source_height % height == 0
        and source_width % width == 0
    )
    if integer_reduction:
        resized = F.interpolate(flat, size=(height, width), mode="area")
    else:
        resized = F.interpolate(
            flat, size=(height, width), mode="bilinear", align_corners=False, antialias=True
        )
    if not source_dtype.is_floating_point:
        info = torch.iinfo(source_dtype)
        resized = resized.round().clamp_(info.min, info.max).to(source_dtype)
    elif resized.dtype != source_dtype:
        resized = resized.to(source_dtype)
    resized = resized.reshape(*lead_shape, channels, height, width)

    if layout == "hwc":
        resized = resized.movedim(-3, -1)
    return resized.numpy() if is_numpy else resized.contiguous()


def _deserialize_stats(stats: dict) -> dict:
    return {
        feature: {name: np.asarray(value) for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def _aggregate_episode_stats(stats_list: list[dict]) -> dict:
    """Aggregate per-episode stats without importing the full dataset stack."""
    if not stats_list:
        raise ValueError("Cannot aggregate an empty list of episode statistics.")
    aggregated = {}
    for feature in (OBS_STATE, ACTION):
        feature_stats = [stats[feature] for stats in stats_list]
        counts = np.asarray(
            [int(stats["count"][0]) for stats in feature_stats], dtype=np.float64
        )
        total_count = counts.sum()
        means = np.stack([stats["mean"] for stats in feature_stats]).astype(np.float64)
        variances = np.stack(
            [np.square(stats["std"]) for stats in feature_stats]
        ).astype(np.float64)
        mean = np.sum(means * counts[:, None], axis=0) / total_count
        variance = (
            np.sum((variances + np.square(means - mean)) * counts[:, None], axis=0)
            / total_count
        )
        aggregated[feature] = {
            "min": np.min(np.stack([stats["min"] for stats in feature_stats]), axis=0),
            "max": np.max(np.stack([stats["max"] for stats in feature_stats]), axis=0),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
            "count": np.asarray([int(total_count)], dtype=np.int64),
        }
    return aggregated


class TeleavatarV2Adapter:
    """Adapt raw TeleAvatar V2 LeRobot samples to the SmolVLA contract."""

    robot_type = "teleavatar"
    source_video_keys = tuple(TELEAVATAR_IMAGE_KEY_MAPPING)

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        adapted_stats: dict | None = None,
        image_hw: dict[str, tuple[int, int]] | None = TELEAVATAR_DEPLOY_IMAGE_HW,
    ):
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        self.adapted_stats = adapted_stats
        # Target (height, width) per SmolVLA camera key after the left-eye crop.
        # Defaults to the deployment feed resolution; `None` disables resizing.
        self.image_hw = dict(image_hw) if image_hw else {}
        if self.adapted_stats is None and self.dataset_root is not None:
            stats_path = self.dataset_root / TELEAVATAR_V2_STATS_PATH
            if stats_path.is_file():
                self.adapted_stats = json.loads(stats_path.read_text())

    def validate_source_features(self, features: dict) -> None:
        for key, indices, expected_names in (
            (OBS_STATE, TELEAVATAR_STATE_INDICES, TELEAVATAR_STATE_NAMES),
            (ACTION, TELEAVATAR_ACTION_INDICES, TELEAVATAR_ACTION_SOURCE_NAMES),
        ):
            if key not in features:
                raise KeyError(
                    f"TeleAvatar V2 metadata is missing required feature {key!r}."
                )
            feature = features[key]
            shape = tuple(feature["shape"])
            if len(shape) != 1 or shape[0] < 48:
                raise ValueError(
                    f"Expected raw TA2 {key!r} shape [D] with D >= 48, got {shape}."
                )
            invalid_indices = [index for index in indices if index >= shape[0]]
            if invalid_indices:
                raise ValueError(
                    f"Invalid TA2 indices {invalid_indices} for {key!r} with shape {shape}."
                )

            names = feature.get("names")
            if isinstance(names, list):
                if len(names) != shape[0]:
                    raise ValueError(
                        f"Feature {key!r} has {len(names)} names but shape {shape}."
                    )
                selected_names = tuple(names[index] for index in indices)
                if selected_names != expected_names:
                    raise ValueError(
                        f"Unexpected TA2 {key!r} schema. Expected selected names {expected_names}, "
                        f"got {selected_names}."
                    )

        missing_images = [
            key for key in TELEAVATAR_IMAGE_KEY_MAPPING if key not in features
        ]
        if missing_images:
            raise KeyError(
                f"TeleAvatar V2 metadata is missing camera features: {missing_images}."
            )

    def _adapt_features(self, features: dict) -> dict:
        self.validate_source_features(features)
        adapted = deepcopy(features)
        adapted[OBS_STATE]["shape"] = (len(TELEAVATAR_STATE_INDICES),)
        adapted[OBS_STATE]["names"] = list(TELEAVATAR_STATE_NAMES)
        adapted[ACTION]["shape"] = (len(TELEAVATAR_ACTION_INDICES),)
        adapted[ACTION]["names"] = list(TELEAVATAR_ACTION_NAMES)

        image_features = []
        for source_key, target_key in TELEAVATAR_IMAGE_KEY_MAPPING.items():
            feature = adapted.pop(source_key)
            shape = tuple(feature["shape"])
            names = feature.get("names")
            if len(shape) != 3:
                raise ValueError(
                    f"Expected camera {source_key!r} to have three dimensions, got {shape}."
                )
            if isinstance(names, list) and names[-1] in ("channel", "channels"):
                height, width, channels = shape
                if width >= 2 * height:
                    width //= 2
                height, width = self.image_hw.get(target_key, (height, width))
                feature["shape"] = (height, width, channels)
            else:
                channels, height, width = shape
                if width >= 2 * height:
                    width //= 2
                height, width = self.image_hw.get(target_key, (height, width))
                feature["shape"] = (channels, height, width)
            image_features.append((target_key, feature))

        # Append in the canonical head, left wrist, right wrist order.
        adapted.update(image_features)
        return adapted

    def _require_adapted_stats(self) -> dict:
        if self.adapted_stats is None:
            expected = (
                self.dataset_root / TELEAVATAR_V2_STATS_PATH
                if self.dataset_root is not None
                else TELEAVATAR_V2_STATS_PATH
            )
            raise FileNotFoundError(
                "TeleAvatar V2 needs statistics computed after state/action adaptation. "
                f"Missing {expected}. Run scripts/compute_teleavatar_v2_stats.py first."
            )
        if self.adapted_stats.get("version") != 1:
            raise ValueError("Unsupported TeleAvatar V2 adapted-stats version.")
        if self.adapted_stats.get("camera_eye") != "left":
            raise ValueError(
                "TeleAvatar V2 adapted stats were not generated for the all-left-eye setup."
            )
        if (
            tuple(self.adapted_stats.get("state_indices", ()))
            != TELEAVATAR_STATE_INDICES
        ):
            raise ValueError(
                "TeleAvatar V2 adapted stats use stale or unexpected state indices. "
                "Regenerate them with scripts/compute_teleavatar_v2_stats.py."
            )
        if (
            tuple(self.adapted_stats.get("action_indices", ()))
            != TELEAVATAR_ACTION_INDICES
        ):
            raise ValueError(
                "TeleAvatar V2 adapted stats use stale or unexpected action indices."
            )
        if self.adapted_stats.get("image_key_mapping") != TELEAVATAR_IMAGE_KEY_MAPPING:
            raise ValueError(
                "TeleAvatar V2 adapted stats use a different camera mapping."
            )
        return self.adapted_stats

    @staticmethod
    def _rename_image_stats(stats: dict) -> dict:
        for source_key, target_key in TELEAVATAR_IMAGE_KEY_MAPPING.items():
            if source_key in stats:
                if target_key in stats:
                    raise KeyError(
                        f"Both source and target camera stats exist for {target_key!r}."
                    )
                stats[target_key] = stats.pop(source_key)
        return stats

    def project_stats(
        self,
        stats: dict | None,
        *,
        copy_stats: bool = False,
        episode_indices: list[int] | None = None,
    ) -> dict | None:
        if stats is None:
            return None

        adapted_stats = self._require_adapted_stats()
        if episode_indices is None:
            replacement = _deserialize_stats(adapted_stats["stats"])
        else:
            episode_stats = []
            for episode_index in episode_indices:
                try:
                    episode_stats.append(
                        _deserialize_stats(
                            adapted_stats["episodes"][str(episode_index)]
                        )
                    )
                except KeyError as exc:
                    raise KeyError(
                        f"Adapted stats are missing episode {episode_index}."
                    ) from exc
            replacement = _aggregate_episode_stats(episode_stats)

        projected = deepcopy(stats) if copy_stats else stats
        projected[OBS_STATE] = replacement[OBS_STATE]
        projected[ACTION] = replacement[ACTION]
        return self._rename_image_stats(projected)

    def adapt_metadata(self, metadata) -> None:
        metadata.info["features"] = self._adapt_features(metadata.info["features"])
        metadata.stats = self.project_stats(metadata.stats)

        adapted_stats = self._require_adapted_stats()
        for episode_index, episode_stats in metadata.episodes_stats.items():
            try:
                replacement = _deserialize_stats(
                    adapted_stats["episodes"][str(episode_index)]
                )
            except KeyError as exc:
                raise KeyError(
                    f"Adapted stats are missing episode {episode_index}."
                ) from exc
            projected = self._rename_image_stats(deepcopy(episode_stats))
            projected[OBS_STATE] = replacement[OBS_STATE]
            projected[ACTION] = replacement[ACTION]
            metadata.episodes_stats[episode_index] = projected

    def adapt_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        if OBS_STATE not in item or ACTION not in item:
            raise KeyError(
                f"TeleAvatar V2 sample must contain {OBS_STATE!r} and {ACTION!r}."
            )
        item[OBS_STATE] = adapt_teleavatar_state(item[OBS_STATE])
        item[ACTION] = adapt_teleavatar_action(item[ACTION])

        for source_key, target_key in TELEAVATAR_IMAGE_KEY_MAPPING.items():
            if source_key in item:
                if target_key in item:
                    raise KeyError(
                        f"Both source and target camera values exist for {target_key!r}."
                    )
                image = item.pop(source_key)
            elif target_key in item:
                image = item[target_key]
            else:
                raise KeyError(
                    f"TeleAvatar V2 sample is missing camera {source_key!r}."
                )
            image = crop_left_stereo_eye(image)
            if target_key in self.image_hw:
                image = resize_image(image, *self.image_hw[target_key])
            item[target_key] = image
        return item


def get_dataset_adapter(
    robot_type: str | None, *, dataset_root: str | Path | None = None
):
    if robot_type == TeleavatarV2Adapter.robot_type:
        return TeleavatarV2Adapter(dataset_root=dataset_root)
    return None
