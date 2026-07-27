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

import logging

from torch import nn

from lerobot.constants import ACTION, OBS_STATE
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
# from lerobot.envs.configs import EnvConfig  # Removed - not needed for SmolVLA2 pretraining
# from lerobot.envs.utils import env_to_policy_features  # Removed - not needed for SmolVLA2 pretraining
# SmolVLA2-only policy factory - removed all other policies
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla2.configuration_smolvla2 import SmolVLA2Config, SmolVLAConfig


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name in {"smolvla", "smolvla2"}:
        from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
        return SmolVLA2Policy
    else:
        raise NotImplementedError(
            f"Policy with name {name} is not implemented. Only SmolVLA/SmolVLA2 is supported."
        )


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "smolvla2":
        return SmolVLA2Config(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    else:
        raise ValueError(
            f"Policy type '{policy_type}' is not available. Only SmolVLA/SmolVLA2 is supported."
        )


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from a dataset
    in order to properly dimension and instantiate a policy for that dataset.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.

    Raises:
        ValueError: ds_meta must be provided.

    Returns:
        PreTrainedPolicy: SmolVLA2 policy instance
    """
    if not ds_meta:
        raise ValueError("Dataset metadata must be provided for SmolVLA2 pretraining.")

    # SmolVLA2-only factory - no backend compatibility checks needed

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    # Parse features from dataset metadata (required for SmolVLA2)
    features = dataset_to_policy_features(ds_meta.features)
    # Single-dataset stats are keyed by feature name. Multi-dataset stats are grouped by robot type.
    if ds_meta.stats and (ACTION in ds_meta.stats or OBS_STATE in ds_meta.stats):
        kwargs["dataset_stats"] = ds_meta.stats
    elif ds_meta.stats:
        aggregated_stats = {}
        for robot_type, stats in ds_meta.stats.items():
            for feature_name, feature_stats in stats.items():
                if feature_name not in aggregated_stats:
                    aggregated_stats[feature_name] = feature_stats
                else:
                    # Statistics are already aggregated per robot type by MultiLeRobotDatasetMeta.
                    # Keep the first compatible feature statistics for the shared policy buffers.
                    pass
        kwargs["dataset_stats"] = aggregated_stats
    else:
        kwargs["dataset_stats"] = ds_meta.stats

    target_output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    target_input_features = {
        key: ft for key, ft in features.items() if key not in target_output_features
    }

    target_state = next(
        (feature for feature in target_input_features.values() if feature.type is FeatureType.STATE),
        None,
    )
    target_action = next(iter(target_output_features.values()), None)
    if target_state is not None and target_state.shape[-1] > cfg.max_state_dim:
        raise ValueError(
            f"Target state dimension {target_state.shape[-1]} exceeds max_state_dim={cfg.max_state_dim}."
        )
    if target_action is not None and target_action.shape[-1] > cfg.max_action_dim:
        raise ValueError(
            f"Target action dimension {target_action.shape[-1]} exceeds max_action_dim={cfg.max_action_dim}."
        )

    if cfg.pretrained_path:
        source_state = cfg.robot_state_feature
        source_action = cfg.action_feature
        logging.info(
            "Adapting pretrained policy features to target dataset: "
            "state=%s->%s, action=%s->%s, cameras=%d->%d",
            source_state.shape[-1] if source_state is not None else None,
            target_state.shape[-1] if target_state is not None else None,
            source_action.shape[-1] if source_action is not None else None,
            target_action.shape[-1] if target_action is not None else None,
            len(cfg.image_features),
            sum(feature.type is FeatureType.VISUAL for feature in target_input_features.values()),
        )

    cfg.output_features = target_output_features
    cfg.input_features = target_input_features
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        logging.info("Initializing policy from pretrained checkpoint: %s", cfg.pretrained_path)
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
