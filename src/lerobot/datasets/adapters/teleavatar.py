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
from copy import deepcopy
from typing import Any

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


class TeleavatarJointPositionAdapter:
    """Project Teleavatar state and action features onto joint-position control space."""

    robot_type = "teleavatar"
    feature_indices = {
        "observation.state": tuple(range(16)),
        "action": tuple(range(16)),
    }

    def _validate_and_project_features(self, features: dict) -> dict:
        projected_features = deepcopy(features)

        for key, indices in self.feature_indices.items():
            if key not in projected_features:
                raise KeyError(f"Teleavatar metadata is missing required feature {key!r}.")

            feature = projected_features[key]
            shape = tuple(feature["shape"])
            if len(shape) != 1:
                raise ValueError(f"Expected {key!r} to have shape [D], got {shape}.")

            original_dim = shape[0]
            invalid_indices = [index for index in indices if index < 0 or index >= original_dim]
            if invalid_indices:
                raise ValueError(
                    f"Invalid indices {invalid_indices} for {key!r} with dimension {original_dim}."
                )

            names = feature.get("names")
            if isinstance(names, list):
                if len(names) != original_dim:
                    raise ValueError(
                        f"Feature {key!r} has {len(names)} names but its shape is {shape}."
                    )
                selected_names = tuple(names[index] for index in indices)
                if selected_names != TELEAVATAR_JOINT_POSITION_NAMES:
                    raise ValueError(
                        f"Unexpected Teleavatar {key!r} schema. Expected the first 16 dimensions to be "
                        f"{TELEAVATAR_JOINT_POSITION_NAMES}, got {selected_names}."
                    )
                feature["names"] = list(selected_names)

            feature["shape"] = (len(indices),)

        return projected_features

    def project_stats(self, stats: dict | None, *, copy_stats: bool = False) -> dict | None:
        if stats is None:
            return None

        projected_stats = deepcopy(stats) if copy_stats else stats
        for key, indices in self.feature_indices.items():
            feature_stats = projected_stats.get(key)
            if feature_stats is None:
                raise KeyError(f"Teleavatar statistics are missing required feature {key!r}.")

            for stat_name in ("mean", "std", "min", "max"):
                value = feature_stats.get(stat_name)
                if value is not None:
                    feature_stats[stat_name] = value[..., list(indices)]

        return projected_stats

    def adapt_metadata(self, metadata) -> None:
        metadata.info["features"] = self._validate_and_project_features(metadata.info["features"])
        metadata.stats = self.project_stats(metadata.stats)
        for episode_index, episode_stats in metadata.episodes_stats.items():
            metadata.episodes_stats[episode_index] = self.project_stats(episode_stats)

    def adapt_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        for key, indices in self.feature_indices.items():
            if key not in item:
                raise KeyError(f"Teleavatar sample is missing required feature {key!r}.")
            item[key] = item[key][..., list(indices)]
        return item


TELEAVATAR_JOINT_POSITION_ADAPTER = TeleavatarJointPositionAdapter()


def get_dataset_adapter(robot_type: str | None):
    if robot_type == TELEAVATAR_JOINT_POSITION_ADAPTER.robot_type:
        return TELEAVATAR_JOINT_POSITION_ADAPTER
    return None
