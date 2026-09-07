"""Checkpoint loading and stateless full-chunk SmolVLA inference."""

from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
import sys
import time

import numpy as np

from .contracts import (
    ACTION_DIM,
    OBS_IMAGE,
    OBS_IMAGE_2,
    OBS_IMAGE_3,
    OBS_STATE,
    STATE_DIM,
    pad_to_width,
    resolve_checkpoint_width,
    validate_state_layout,
)


def load_policy_config(checkpoint_path: str | Path):
    """Load either repository-native or official SmolVLA checkpoint config."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla2.configuration_smolvla2 import (
        SmolVLA2Config,
        SmolVLAConfig,
    )

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    if not isinstance(config, (SmolVLAConfig, SmolVLA2Config)):
        raise TypeError(
            "Deployment supports only SmolVLA/SmolVLA2 checkpoints, "
            f"got policy type '{config.type}'."
        )
    return config


class SmolVLARuntime:
    """Load one checkpoint and infer complete chunks without an action queue."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        smolvla_repo: str | Path,
        device: str = "cuda",
        use_amp: bool | None = None,
        vlm_model_path: str | Path | None = None,
    ) -> None:
        repo = Path(smolvla_repo).expanduser().resolve()
        source = repo / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"SmolVLA source directory not found: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        import torch
        from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not (checkpoint_path / "config.json").is_file():
            raise FileNotFoundError(f"Checkpoint config.json not found in {checkpoint_path}")
        if not (checkpoint_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Checkpoint model.safetensors not found in {checkpoint_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

        config = load_policy_config(checkpoint_path)
        config.device = device
        if vlm_model_path is not None:
            local_vlm = Path(vlm_model_path).expanduser().resolve()
            if not local_vlm.is_dir():
                raise FileNotFoundError(f"Local VLM model directory not found: {local_vlm}")
            config.vlm_model_name = str(local_vlm)
        # A deployment checkpoint contains the complete SmolVLA policy state.
        # Loading the base VLM weights here would be redundant because
        # SmolVLA2Policy.from_pretrained() restores model.safetensors
        # immediately after constructing the architecture.  It also makes
        # deployment depend unnecessarily on the base snapshot's weight
        # shards; only its config, tokenizer, and processors are required.
        if config.load_vlm_weights:
            logging.info(
                "Disabling redundant base-VLM weight preload; restoring the complete policy from %s",
                checkpoint_path / "model.safetensors",
            )
            config.load_vlm_weights = False
        self._state_width, self._action_width = self._validate_config(config)
        self._torch = torch
        self._device = torch.device(device)
        self._use_amp = config.use_amp if use_amp is None else bool(use_amp)
        # (height, width) each camera was trained at, from the checkpoint's
        # input features (stored channel-first). The training adapter downscales
        # dataset frames to the RTP feed resolution, so a mismatch here means the
        # policy's 512x512 resize is starting from a different source than it
        # saw in training. Reported once per camera rather than per frame.
        self._trained_image_hw = {
            key: tuple(int(v) for v in feature.shape[-2:])
            for key, feature in config.image_features.items()
        }
        self._image_hw_warned: set[str] = set()

        logging.info("Loading SmolVLA checkpoint from %s", checkpoint_path)
        self.policy = SmolVLA2Policy.from_pretrained(
            checkpoint_path,
            config=config,
            local_files_only=True,
        )
        self.policy.to(self._device)
        self.policy.eval()
        self._validate_normalization_statistics(self.policy)
        self.chunk_size = int(config.chunk_size)
        logging.info(
            "Loaded SmolVLA: device=%s chunk_size=%d action_dim=%d (checkpoint stores state=%d action=%d)",
            self._device,
            self.chunk_size,
            ACTION_DIM,
            self._state_width,
            self._action_width,
        )

    @staticmethod
    def _validate_config(config) -> tuple[int, int]:
        """Check the checkpoint contract and return its stored feature widths."""
        validate_state_layout(getattr(config, "teleavatar_state_layout", None))
        if config.robot_state_feature is None:
            raise ValueError("Checkpoint has no observation.state feature")
        if config.action_feature is None:
            raise ValueError("Checkpoint has no action feature")
        state_width = resolve_checkpoint_width(
            OBS_STATE,
            config.robot_state_feature.shape,
            STATE_DIM,
            getattr(config, "max_state_dim", None),
        )
        action_width = resolve_checkpoint_width(
            "action",
            config.action_feature.shape,
            ACTION_DIM,
            getattr(config, "max_action_dim", None),
        )
        required_images = {OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3}
        missing = required_images.difference(config.image_features)
        if missing:
            raise ValueError(f"Checkpoint is missing TeleAvatar camera features: {sorted(missing)}")
        if config.n_obs_steps != 1:
            raise ValueError("This deployment runner currently requires n_obs_steps=1")
        if config.predict_relative_actions:
            raise ValueError("This checkpoint predicts relative actions; absolute TeleAvatar actions are required")
        if config.adapt_to_pi_aloha:
            raise ValueError("adapt_to_pi_aloha must be disabled for TeleAvatar")
        return state_width, action_width

    @staticmethod
    def _validate_normalization_statistics(policy) -> None:
        prefixes = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
        invalid = []
        normalization_tensors = 0
        for named_tensors in (policy.named_parameters(), policy.named_buffers()):
            for name, value in named_tensors:
                if not name.startswith(prefixes) or not value.is_floating_point():
                    continue
                normalization_tensors += 1
                if not value.isfinite().all().item():
                    invalid.append(name)
        if invalid:
            raise RuntimeError(
                "Checkpoint normalization statistics contain infinity or NaN after loading: "
                + ", ".join(sorted(invalid))
            )
        if normalization_tensors == 0:
            logging.warning("Policy has no non-identity normalization statistics to validate")

    def _make_batch(self, observation: dict, task: str) -> dict:
        torch = self._torch
        state = np.asarray(observation[OBS_STATE], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected state ({STATE_DIM},), got {state.shape}")
        # A checkpoint trained on several datasets normalizes a padded state, so
        # the buffers are `max_state_dim` wide and a bare 14-D tensor would not
        # broadcast against them.
        state = pad_to_width(state, self._state_width)
        batch = {
            OBS_STATE: torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(self._device),
            "task": task,
        }
        for key in (OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3):
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC RGB image for {key}, got {image.shape}")
            trained_hw = self._trained_image_hw.get(key)
            if (
                trained_hw is not None
                and tuple(image.shape[:2]) != trained_hw
                and key not in self._image_hw_warned
            ):
                self._image_hw_warned.add(key)
                logging.warning(
                    "%s arrives at %dx%d but the checkpoint was trained at %dx%d; "
                    "the policy resizes both to 512x512, but the source resolution "
                    "differs from training (check the RTP split regions or retrain "
                    "with the matching TELEAVATAR_DEPLOY_IMAGE_HW).",
                    key,
                    image.shape[0],
                    image.shape[1],
                    trained_hw[0],
                    trained_hw[1],
                )
            tensor = torch.from_numpy(np.ascontiguousarray(image))
            tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
            batch[key] = tensor.unsqueeze(0).to(self._device)
        return batch

    def infer_action_chunk(self, observation: dict, task: str) -> tuple[np.ndarray, float]:
        """Return unnormalized direct-trigger actions with shape [T, 16]."""
        torch = self._torch
        batch = self._make_batch(observation, task)
        amp = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._use_amp
            else nullcontext()
        )
        start = time.monotonic()
        with torch.inference_mode(), amp:
            normalized = self.policy.normalize_inputs(batch)
            images, image_masks = self.policy.prepare_images(normalized)
            state = self.policy.prepare_state(normalized)
            language_tokens, language_masks = self.policy.prepare_language(normalized)
            actions = self.policy.model.sample_actions(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state,
            )
            # `sample_actions` always emits `max_action_dim` columns, so narrow
            # to whatever width the unnormalize buffers were saved at before
            # applying them, and only then down to the real TeleAvatar action.
            actions = actions[:, :, : self._action_width]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
            actions = actions[:, :, :ACTION_DIM]
        elapsed_ms = (time.monotonic() - start) * 1000.0
        chunk = actions[0].detach().to(dtype=torch.float32, device="cpu").numpy()
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not np.isfinite(chunk).all():
            raise RuntimeError(f"Policy returned an invalid action chunk: {chunk.shape}")
        return chunk, elapsed_ms
