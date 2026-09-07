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
import os
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(cfg: TrainPipelineConfig, return_list: bool = False) -> list[str] | str:
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.type}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        # Create shorter dataset tag to avoid 64-char limit
        repo_id = cfg.dataset.repo_id
        if "," in repo_id:
            # Multiple datasets - use count
            dataset_count = len(repo_id.split(","))
            lst.append(f"datasets:{dataset_count}")
        else:
            # Single dataset - use last part of path
            dataset_name = repo_id.split("/")[-1][:20]  # Truncate to 20 chars
            lst.append(f"dataset:{dataset_name}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    return lst if return_list else "-".join(lst)


def get_safe_trackio_artifact_name(name: str):
    """TrackIO artifacts don't accept ":" or "/" in their name."""
    return name.replace(":", "_").replace("/", "_")


class TrackIOLogger:
    """A helper class to log objects using TrackIO."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.trackio
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # Set up TrackIO with wandb-compatible API
        try:
            import trackio as wandb  # TrackIO provides wandb-compatible API
        except ImportError:
            raise ImportError(
                "TrackIO is not installed. Please install it with: pip install trackio"
            )

        # Initialize TrackIO with wandb-compatible API
        # Note: TrackIO doesn't support all wandb parameters, so we filter them
        init_kwargs = {
            "project": self.cfg.project,
            "name": self.job_name,
            "config": cfg.to_dict(),
        }
        
        # Add optional parameters if they exist and are supported
        # Note: TrackIO doesn't support 'notes' parameter in init()
        # if self.cfg.notes:
        #     init_kwargs["notes"] = self.cfg.notes
            
        # Add TrackIO-specific space_id if provided
        if hasattr(self.cfg, 'space_id') and self.cfg.space_id:
            init_kwargs["space_id"] = self.cfg.space_id
            
        wandb.init(**init_kwargs)
        
        # Handle custom step key for rl asynchronous training
        self._trackio_custom_step_key: set[str] | None = None
        print(colored("Logs will be tracked with TrackIO.", "blue", attrs=["bold"]))
        logging.info(colored("TrackIO logging initialized successfully", 'green', attrs=['bold']))
        self._trackio = wandb

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to TrackIO (local storage)."""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_trackio_artifact_name(artifact_name)
        
        # TrackIO stores artifacts locally, so we just log the path
        checkpoint_path = checkpoint_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE
        if checkpoint_path.exists():
            self._trackio.log({
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": step_id,
                "artifact_name": artifact_name
            })
            logging.info(f"Logged checkpoint artifact: {artifact_name}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # Handle custom step keys similar to WandB implementation
        if custom_step_key is not None:
            if self._trackio_custom_step_key is None:
                self._trackio_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._trackio_custom_step_key:
                self._trackio_custom_step_key.add(new_custom_key)

        # Collect all valid metrics to log in a single batch
        metrics_to_log = {}
        for k, v in d.items():
            if not isinstance(v, (int, float, str)):
                logging.warning(
                    f'TrackIO logging of key "{k}" was ignored as its type "{type(v)}" is not handled by this wrapper.'
                )
                continue

            # Do not log the custom step key itself
            if self._trackio_custom_step_key is not None and k in self._trackio_custom_step_key:
                continue

            metrics_to_log[f"{mode}/{k}"] = v

        # Add custom step key if provided
        if custom_step_key is not None:
            metrics_to_log[f"{mode}/{custom_step_key}"] = d[custom_step_key]

        # Log all metrics in a single call
        if metrics_to_log:
            self._trackio.log(metrics_to_log, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """Log video to TrackIO."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        # TrackIO handles video logging through file paths
        self._trackio.log({
            f"{mode}/video_path": video_path,
            f"{mode}/video_step": step,
            f"{mode}/video_fps": self.env_fps
        }, step=step)
        
    def finish(self):
        """Finish the TrackIO run."""
        if hasattr(self._trackio, 'finish'):
            self._trackio.finish()
