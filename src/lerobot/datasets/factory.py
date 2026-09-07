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
import random
from pathlib import Path
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.adapters import get_dataset_adapter

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from lerobot.datasets.utils_must import EPISODES_DATASET_MAPPING, FEATURE_KEYS_MAPPING


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def split_train_val_episodes(
    base_episodes: list[int] | None,
    total_episodes: int,
    num_val: int,
    seed: int,
    name: str,
) -> tuple[list[int] | None, list[int]]:
    """Hold out whole episodes for validation, deterministically.

    The split must be by episode, not by frame: consecutive frames of the same
    episode are near-duplicates at 30 fps, so a frame-level holdout scores the
    model on data it has effectively trained on.

    The RNG is seeded from `seed` and `name` (the dataset's repo_id) rather than
    from the dataset's position in the run, so adding or reordering datasets in a
    multi-dataset run leaves the holdout of every other dataset untouched.

    Returns `(train_episodes, val_episodes)`. When the holdout is disabled or
    impossible, `train_episodes` is `base_episodes` unchanged -- including
    `None`, which means "every episode" -- so training behaves exactly as it did
    before this option existed.
    """
    if num_val <= 0:
        return base_episodes, []

    all_eps = list(base_episodes) if base_episodes is not None else list(range(total_episodes))
    # Never give validation more than half of a dataset, and always leave at
    # least one episode behind to train on.
    n_val = min(num_val, len(all_eps) // 2)
    if n_val <= 0:
        logging.warning(
            f"Dataset '{name}' has {len(all_eps)} episode(s); too few to hold any out for "
            "validation. It contributes to training only."
        )
        return base_episodes, []

    rng = random.Random(f"{seed}:{name}")
    val_eps = sorted(rng.sample(all_eps, n_val))
    val_set = set(val_eps)
    train_eps = [ep for ep in all_eps if ep not in val_set]
    return train_eps, val_eps


def make_dataset(
    cfg: TrainPipelineConfig, split: str = "train"
) -> LeRobotDataset | MultiLeRobotDataset | None:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.
        split (str): "train" for the episodes to fit on, "val" for the held-out
            episodes described by `cfg.dataset.val_episodes_per_dataset`.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset, or None for `split="val"` when no
        episodes were held out.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}.")

    # Validation must see the images the deployed policy sees, so the random
    # train-time augmentations are switched off for it.
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable and split == "train"
        else None
    )
    num_val_episodes = cfg.dataset.val_episodes_per_dataset
    val_split_seed = cfg.dataset.val_split_seed

    if "," in cfg.dataset.repo_id:
        repo_id = cfg.dataset.repo_id.split(",")
        repo_id = [r for r in repo_id if r]
    else:
        repo_id = cfg.dataset.repo_id
    sampling_weights = cfg.dataset.sampling_weights.split(",") if cfg.dataset.sampling_weights else None
    feature_keys_mapping = FEATURE_KEYS_MAPPING
    if isinstance(repo_id, str):
        revision = getattr(cfg.dataset, "revision", None)
        root = getattr(cfg.dataset, "root", None)
        # If root is provided, construct the full path as root/repo_id
        dataset_root = Path(root) / cfg.dataset.repo_id if root else None
        # If root is provided, use local_files_only=True to prevent downloads from HuggingFace
        local_files_only = root is not None
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id,
            root=dataset_root,
            feature_keys_mapping=feature_keys_mapping,
            revision=revision,
            local_files_only=local_files_only,
        )
        feature_adapter = get_dataset_adapter(ds_meta.robot_type, dataset_root=ds_meta.root)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        train_episodes, val_episodes = split_train_val_episodes(
            cfg.dataset.episodes,
            ds_meta.total_episodes,
            num_val_episodes,
            val_split_seed,
            cfg.dataset.repo_id,
        )
        if split == "val":
            if not val_episodes:
                return None
            logging.info(f"Validation holdout for '{cfg.dataset.repo_id}': episodes {val_episodes}")
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=dataset_root,
            episodes=val_episodes if split == "val" else train_episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            revision=revision,
            video_backend=cfg.dataset.video_backend,
            download_videos=True,
            local_files_only=local_files_only,
            feature_keys_mapping=feature_keys_mapping,
            max_action_dim=cfg.dataset.max_action_dim,
            max_state_dim=cfg.dataset.max_state_dim,
            max_num_images=cfg.dataset.max_num_images,
            max_image_dim=cfg.dataset.max_image_dim,
            feature_adapter=feature_adapter,
        )
    else:
        delta_timestamps = {}
        episodes = {}
        feature_adapters = {}
        root = getattr(cfg.dataset, "root", None)
        # If root is provided, use local_files_only=True to prevent downloads from HuggingFace

        local_files_only = root is not None
        for i in range(len(repo_id)):
            # For multi-dataset with root, each dataset is at root/repo_id[i]
            dataset_root = Path(root) / repo_id[i] if root else None

            ds_meta = LeRobotDatasetMetadata(
                repo_id[i],
                root=dataset_root,
                feature_keys_mapping=feature_keys_mapping,
                local_files_only=local_files_only,
            )  # FIXME(mshukor): ?
            delta_timestamps[repo_id[i]] = resolve_delta_timestamps(cfg.policy, ds_meta)
            base_episodes = EPISODES_DATASET_MAPPING.get(repo_id[i], cfg.dataset.episodes)
            train_episodes, val_episodes = split_train_val_episodes(
                base_episodes,
                ds_meta.total_episodes,
                num_val_episodes,
                val_split_seed,
                repo_id[i],
            )
            episodes[repo_id[i]] = val_episodes if split == "val" else train_episodes
            feature_adapters[repo_id[i]] = get_dataset_adapter(
                ds_meta.robot_type, dataset_root=ds_meta.root
            )

        if split == "val":
            # A dataset too small to give up an episode contributes nothing here.
            # It has to be dropped rather than passed an empty episode list, and
            # `sampling_weights` is positional, so it is filtered in lockstep.
            keep = [i for i in range(len(repo_id)) if episodes[repo_id[i]]]
            if not keep:
                return None
            repo_id = [repo_id[i] for i in keep]
            if sampling_weights is not None:
                sampling_weights = [sampling_weights[i] for i in keep]
            for r in repo_id:
                logging.info(f"Validation holdout for '{r}': episodes {episodes[r]}")
        # training_features = TRAINING_FEATURES.get(cfg.dataset.features_version, None)
        # FIXME: (jadechoghari): check support for training features
        training_features = None
        dataset = MultiLeRobotDataset(
            repo_id,
            root=root,
            # TODO(aliberts): add proper support for multi dataset
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
            download_videos=True,
            local_files_only=local_files_only,
            sampling_weights=sampling_weights,
            feature_keys_mapping=feature_keys_mapping,
            max_action_dim=cfg.policy.max_action_dim,
            max_state_dim=cfg.policy.max_state_dim,
            max_num_images=cfg.dataset.max_num_images,
            max_image_dim=cfg.dataset.max_image_dim,
            train_on_all_features=cfg.dataset.train_on_all_features,
            training_features=training_features,
            discard_first_n_frames=cfg.dataset.discard_first_n_frames,
            min_fps=cfg.dataset.min_fps,
            max_fps=cfg.dataset.max_fps,
            discard_first_idle_frames=cfg.dataset.discard_first_idle_frames,
            motion_threshold=cfg.dataset.motion_threshold,
            motion_window_size=cfg.dataset.motion_window_size,
            motion_buffer=cfg.dataset.motion_buffer,
            feature_adapters=feature_adapters,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )
    if cfg.dataset.use_imagenet_stats:
        # Initialize stats structure if it doesn't exist
        if dataset.meta.stats is None:
            dataset.meta.stats = {}
        
        for key in getattr(dataset, 'camera_keys', []):
            # Initialize stats for this camera key if it doesn't exist
            if key not in dataset.meta.stats or dataset.meta.stats[key] is None:
                dataset.meta.stats[key] = {}
                
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
