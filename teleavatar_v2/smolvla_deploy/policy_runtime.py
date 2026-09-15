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
    """Load one checkpoint and infer complete chunks without an action queue.

    On CUDA the whole inference - uint8 frames in, unnormalized action chunk
    out - is captured once into a CUDA graph and replayed afterwards. The
    eager path launches a few thousand small kernels from Python per chunk, and
    every launch has to reacquire the GIL; with the decoder and ROS2 callbacks
    running in the same process that turned an 86ms inference into 150-280ms
    with large jitter. A replay is one launch, so it is immune to that.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        smolvla_repo: str | Path,
        device: str = "cuda",
        use_amp: bool | None = None,
        num_steps: int | None = None,
        profile_inference: bool = False,
        vlm_model_path: str | Path | None = None,
        precision: str | None = None,
        attn_implementation: str = "sdpa",
        compile_model: bool = False,
        cuda_graph: bool = True,
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
        if num_steps is not None:
            if num_steps <= 0:
                raise ValueError("num_steps must be positive")
            logging.info("Overriding flow denoising steps: %d -> %d", config.num_steps, num_steps)
            config.num_steps = int(num_steps)
        # The hand-rolled attention in `smolvlm_with_expert2` defaults to an eager
        # kernel that materializes the full [B, H, Lq, Lk] score matrix in fp32.
        # Training keeps that for reproducibility; deployment has no such reason.
        if attn_implementation not in ("eager", "sdpa"):
            raise ValueError(f"attn-implementation must be 'eager' or 'sdpa', got '{attn_implementation}'")
        config.attn_implementation = attn_implementation
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
        self._profile_inference = bool(profile_inference)
        self._compute_dtype = self._resolve_precision(torch, precision, self._device)
        if self._compute_dtype is not None and self._use_amp:
            # Holding the weights in a narrow dtype already gives what autocast
            # was for, and autocast on top would re-cast every fp32 weight it
            # sees on each Linear call.
            logging.info("Disabling autocast: the backbone already runs in %s", self._compute_dtype)
            self._use_amp = False
        self._language_cache: dict[tuple[str, int, str], tuple[object, object]] = {}
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
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
        if self._compute_dtype is not None:
            # Only the backbone. The normalization buffers stay fp32: they are
            # applied outside `policy.model` and mean/std in bf16 would quantize
            # the joint statistics the action chunk is unnormalized with.
            self.policy.model.to(dtype=self._compute_dtype)
        self._validate_normalization_statistics(self.policy)
        self._cuda_graph_enabled = bool(cuda_graph) and self._device.type == "cuda"
        self._graph: _CapturedInference | None = None
        if compile_model and self._cuda_graph_enabled:
            # A replay has no Python between kernels, which is all the compiled
            # denoise step was buying; and a capture over dynamo-generated code
            # fails on this stack. The graph wins on both counts.
            logging.info("Ignoring --compile: the CUDA graph already removes the launch overhead")
            compile_model = False
        if compile_model:
            self._compile_denoise_step()
        self.chunk_size = int(config.chunk_size)
        _ckpt_dtype = getattr(config, "model_dtype", "fp32(mixed-legacy)")
        logging.info(
            "Loaded SmolVLA: device=%s dtype=%s attn=%s cuda_graph=%s chunk_size=%d num_steps=%d "
            "action_dim=%d (checkpoint stores state=%d action=%d)",
            self._device,
            self._compute_dtype or _ckpt_dtype,
            attn_implementation,
            self._cuda_graph_enabled,
            self.chunk_size,
            int(config.num_steps),
            ACTION_DIM,
            self._state_width,
            self._action_width,
        )

    @staticmethod
    def _resolve_precision(torch, precision: str | None, device) -> object | None:
        """Return the dtype to cast the model to after loading, or None to leave it as built.

        None (the default) means no conversion: the checkpoint is loaded exactly as
        stored, preserving the dtype layout it was trained with.  New checkpoints
        built with model_dtype="bf16" will be uniformly bfloat16; old mixed-dtype
        checkpoints will stay mixed.  Either way the inference numerics match
        training without any explicit flag.

        Pass an explicit precision only to override that layout intentionally, e.g.
        to test numerics or squeeze out latency.  "fp32" forces everything to fp32;
        "bf16" and "fp16" unify to those dtypes — useful only if the checkpoint was
        saved uniformly (model_dtype="bf16") and you want to confirm you are not
        accidentally hitting fp32 paths.
        """
        if precision is None:
            return None
        choices = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}
        if precision not in choices:
            raise ValueError(f"precision must be one of {sorted(choices)}, got '{precision}'")
        dtype = choices[precision]
        if dtype is None:
            return None
        if device.type != "cuda":
            logging.warning(
                "precision=%s requested on %s; leaving weight dtypes as built", precision, device.type
            )
            return None
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logging.warning(
                "This GPU has no bfloat16 support; leaving weight dtypes as built "
                "(try --precision fp16)"
            )
            return None
        return dtype

    def _compile_denoise_step(self) -> None:
        """Compile the per-step expert pass, which is what the loop repeats.

        `sample_actions` is not `forward`, so wrapping the module would leave it
        interpreted; the denoise step is where `num_steps` x `num_vlm_layers`
        worth of small kernel launches actually accumulate. Dynamo errors are
        suppressed so a compile failure degrades to eager instead of taking the
        robot down mid-run, and the first chunk pays the compilation.
        """
        torch = self._torch
        try:
            import torch._dynamo as dynamo

            dynamo.config.suppress_errors = True
        except ImportError:  # pragma: no cover - torch always ships dynamo in 2.x
            logging.warning("torch._dynamo unavailable; skipping compilation")
            return
        model = self.policy.model
        model.denoise_step = torch.compile(model.denoise_step, dynamic=False)
        logging.info("Compiled denoise_step; the first chunk will include compilation time")

    def _synchronize_for_profile(self) -> None:
        if self._profile_inference and self._device.type == "cuda":
            self._torch.cuda.synchronize(self._device)

    def _profile_mark(self, marks: list[tuple[str, float]], name: str) -> None:
        if self._profile_inference:
            self._synchronize_for_profile()
            marks.append((name, time.monotonic()))

    def _prepare_language_cached(self, batch: dict, task: str):
        batch_size = int(batch[OBS_STATE].shape[0])
        cache_key = (task, batch_size, str(self._device))
        cached = self._language_cache.get(cache_key)
        if cached is not None:
            return cached
        language = self.policy.prepare_language(batch)
        self._language_cache[cache_key] = language
        return language

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

    def _validate_observation(self, observation: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Check shapes and return the padded state and the HWC frames, still on the host."""
        state = np.asarray(observation[OBS_STATE], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected state ({STATE_DIM},), got {state.shape}")
        # A checkpoint trained on several datasets normalizes a padded state, so
        # the buffers are `max_state_dim` wide and a bare 14-D tensor would not
        # broadcast against them.
        state = np.ascontiguousarray(pad_to_width(state, self._state_width))
        images: dict[str, np.ndarray] = {}
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
            images[key] = np.ascontiguousarray(image)
        return state, images

    def _make_batch(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> dict:
        torch = self._torch
        batch = {
            OBS_STATE: torch.from_numpy(state).unsqueeze(0).to(self._device),
            "task": task,
        }
        for key, image in images.items():
            tensor = torch.from_numpy(image)
            if tensor.dtype == torch.uint8:
                # Send the 8-bit frame across PCIe and widen it on the GPU: a
                # quarter of the bytes of an fp32 conversion done host-side, for
                # three cameras on every control tick. Bit-identical result.
                tensor = tensor.to(self._device, non_blocking=True)
                tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
            else:
                tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
                tensor = tensor.to(self._device)
            batch[key] = tensor.unsqueeze(0)
        return batch

    def _finish_chunk(self, chunk: np.ndarray, marks: list[tuple[str, float]]) -> tuple[np.ndarray, float]:
        elapsed_ms = (time.monotonic() - marks[0][1]) * 1000.0
        if self._profile_inference:
            parts = []
            for (prev_name, prev_time), (name, current_time) in zip(
                marks[:-1], marks[1:], strict=True
            ):
                parts.append(f"{prev_name}->{name}={(current_time - prev_time) * 1000.0:.1f}ms")
            logging.info("inference profile: %s total=%.1fms", " ".join(parts), elapsed_ms)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or not np.isfinite(chunk).all():
            raise RuntimeError(f"Policy returned an invalid action chunk: {chunk.shape}")
        return chunk, elapsed_ms

    def infer_action_chunk(self, observation: dict, task: str) -> tuple[np.ndarray, float]:
        """Return unnormalized direct-trigger actions with shape [T, 16]."""
        marks: list[tuple[str, float]] = [("start", time.monotonic())]
        state, images = self._validate_observation(observation)
        self._profile_mark(marks, "validate")
        if self._cuda_graph_enabled and all(image.dtype == np.uint8 for image in images.values()):
            graph = self._captured_inference(state, images, task)
            return self._finish_chunk(graph.run(state, images, marks), marks)
        return self._finish_chunk(self._infer_eager(state, images, task, marks), marks)

    def _infer_eager(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        marks: list[tuple[str, float]],
        noise=None,
    ) -> np.ndarray:
        """Kernel-by-kernel inference; `noise` overrides the sampled flow noise (tests, graph check)."""
        torch = self._torch
        batch = self._make_batch(state, images, task)
        self._profile_mark(marks, "make_batch")
        amp = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._use_amp
            else nullcontext()
        )
        with torch.inference_mode(), amp:
            normalized = self.policy.normalize_inputs(batch)
            self._profile_mark(marks, "normalize")
            images, image_masks = self.policy.prepare_images(normalized)
            self._profile_mark(marks, "images")
            state = self.policy.prepare_state(normalized)
            self._profile_mark(marks, "state")
            language_tokens, language_masks = self._prepare_language_cached(normalized, task)
            self._profile_mark(marks, "language")
            actions = self.policy.model.sample_actions(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state,
                noise=None if noise is None else noise.clone(),
            )
            self._profile_mark(marks, "sample")
            # `sample_actions` always emits `max_action_dim` columns, so narrow
            # to whatever width the unnormalize buffers were saved at before
            # applying them, and only then down to the real TeleAvatar action.
            actions = actions[:, :, : self._action_width]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]
            actions = actions[:, :, :ACTION_DIM]
        chunk = actions[0].detach().to(dtype=torch.float32, device="cpu").numpy()
        self._profile_mark(marks, "cpu")
        return chunk

    # --- CUDA graph path ----------------------------------------------------

    def _captured_inference(
        self, state: np.ndarray, images: dict[str, np.ndarray], task: str
    ) -> "_CapturedInference":
        """The graph for these input shapes and task, capturing it first if needed."""
        signature = (task, tuple((key, image.shape) for key, image in images.items()))
        graph = self._graph
        if graph is not None and graph.signature == signature:
            return graph
        if graph is not None:
            logging.warning("Observation shapes or task changed; re-capturing the inference graph")
            self._graph = None
        try:
            self._graph = _CapturedInference(self, signature, state, images, task)
        except Exception as error:
            # Not recoverable in-process: an aborted capture leaves the CUDA RNG
            # registered to a graph that never finished, and the eager path
            # then fails on its first `torch.normal`. Failing here happens in
            # the runner's warm-up, before anything is published.
            raise RuntimeError(
                "CUDA graph capture failed. Re-run with --no-cuda-graph to use the eager path, "
                "which is several times slower under load."
            ) from error
        return self._graph

    def _graph_forward(
        self, images_u8: dict[str, object], state: object, language: tuple[object, object], noise: object
    ) -> object:
        """Device frames and state in, [chunk_size, ACTION_DIM] fp32 actions out.

        The same maths as `_infer_eager`, minus everything that would break a
        capture: the host-side `isinf` asserts inside the normalization
        modules and the pageable host-to-device copies of `_make_batch`.
        """
        torch = self._torch
        policy = self.policy
        batch = {OBS_STATE: state}
        for key, tensor in images_u8.items():
            batch[key] = (
                tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0).unsqueeze(0)
            )
        batch = _apply_normalization(policy.normalize_inputs, batch, inverse=False)
        # Same autocast decision as the eager path (a checkpoint with use_amp
        # runs its fp32 projections in bf16 there too). The cast cache is off
        # because it would pin weight copies allocated inside the capture.
        amp = (
            torch.autocast(device_type=self._device.type, cache_enabled=False)
            if self._use_amp
            else nullcontext()
        )
        with amp:
            images, image_masks = policy.prepare_images(batch)
            prepared_state = policy.prepare_state(batch)
            language_tokens, language_masks = language
            # `sample_actions` integrates in place on the tensor it is handed, so
            # give it a copy and keep the static noise buffer as a pure input.
            actions = policy.model.sample_actions(
                images, image_masks, language_tokens, language_masks, prepared_state, noise=noise.clone()
            )
            actions = actions[:, :, : self._action_width]
            actions = _apply_normalization(policy.unnormalize_outputs, {"action": actions}, inverse=True)["action"]
        return actions[0, :, :ACTION_DIM].to(dtype=torch.float32)


def _apply_normalization(module, batch: dict, *, inverse: bool) -> dict:
    """`Normalize.forward` / `Unnormalize.forward` without their host-synchronizing asserts.

    The statistics were already checked for infinities when the checkpoint was
    loaded (`_validate_normalization_statistics`), and a CUDA graph cannot
    contain the `.any()` those asserts evaluate on the host.
    """
    from lerobot.configs.types import NormalizationMode

    batch = dict(batch)
    for key, feature in module.features.items():
        if key not in batch:
            continue
        mode = module.norm_map.get(feature.type, NormalizationMode.IDENTITY)
        if mode is NormalizationMode.IDENTITY:
            continue
        buffer = getattr(module, "buffer_" + key.replace(".", "_"))
        if mode is NormalizationMode.MEAN_STD:
            mean, std = buffer["mean"], buffer["std"]
            if inverse:
                batch[key] = batch[key] * std + mean
            else:
                batch[key] = (batch[key] - mean) / (std + 1e-8)
        elif mode is NormalizationMode.MIN_MAX:
            low, high = buffer["min"], buffer["max"]
            if inverse:
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (high - low) + low
            else:
                batch[key] = (batch[key] - low) / (high - low + 1e-8)
                batch[key] = batch[key] * 2 - 1
        else:
            raise ValueError(mode)
    return batch


class _CapturedInference:
    """One CUDA graph over `SmolVLARuntime._graph_forward` with static I/O buffers."""

    # Warm-up passes on a side stream before capture: cuBLAS/cuDNN allocate
    # their workspaces and pick kernels lazily, which is not capturable.
    WARMUP_PASSES = 3
    # The graph must reproduce the eager path. Actions are joint angles in
    # radians and the two paths run the same kernels, so anything visible here
    # is a logic error, not rounding.
    MAX_ABS_DIFFERENCE = 1e-3

    def __init__(
        self,
        runtime: SmolVLARuntime,
        signature: tuple,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
    ) -> None:
        torch = runtime._torch
        self._torch = torch
        self._runtime = runtime
        self._device = runtime._device
        self.signature = signature
        started = time.monotonic()

        # Pinned staging buffers make the per-chunk uploads asynchronous DMA
        # instead of the synchronous pageable copies `torch.from_numpy(...).to()` does.
        self._host_images = {
            key: torch.empty(image.shape, dtype=torch.uint8).pin_memory() for key, image in images.items()
        }
        self._device_images = {
            key: torch.empty(image.shape, dtype=torch.uint8, device=self._device)
            for key, image in images.items()
        }
        self._host_state = torch.empty((1, state.shape[0]), dtype=torch.float32).pin_memory()
        self._device_state = torch.empty((1, state.shape[0]), dtype=torch.float32, device=self._device)
        config = runtime.policy.config
        self._noise = torch.empty(
            (1, int(config.chunk_size), int(config.max_action_dim)), dtype=torch.float32, device=self._device
        )
        self._upload(state, images)
        self._noise.normal_()
        language = runtime._prepare_language_cached({OBS_STATE: self._device_state, "task": task}, task)

        forward = lambda: runtime._graph_forward(  # noqa: E731
            self._device_images, self._device_state, language, self._noise
        )
        side_stream = torch.cuda.Stream(device=self._device)
        side_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(self.WARMUP_PASSES):
                forward()
        torch.cuda.current_stream(self._device).wait_stream(side_stream)
        torch.cuda.synchronize(self._device)

        # "thread_local": only this thread's CUDA calls are checked. The default
        # "global" mode invalidates the capture when any other thread in the
        # process touches CUDA, and the H.265 decoder does exactly that from
        # the GStreamer streaming thread throughout.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, capture_error_mode="thread_local"), torch.inference_mode():
            self._device_output = forward()
        torch.cuda.synchronize(self._device)

        self._verify_against_eager(state, images, task)
        logging.info(
            "Captured the inference CUDA graph in %.1fs: images=%s state=%d",
            time.monotonic() - started,
            ", ".join(f"{key.rsplit('.', 1)[-1]}={tuple(image.shape)}" for key, image in images.items()),
            state.shape[0],
        )

    def _upload(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        torch = self._torch
        for key, image in images.items():
            self._host_images[key].copy_(torch.from_numpy(image))
            self._device_images[key].copy_(self._host_images[key], non_blocking=True)
        self._host_state[0].copy_(torch.from_numpy(state))
        self._device_state.copy_(self._host_state, non_blocking=True)

    def _verify_against_eager(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> None:
        """Replay once and compare with the eager path on the same noise."""
        runtime = self._runtime
        expected = runtime._infer_eager(state, images, task, [("start", time.monotonic())], noise=self._noise)
        self._graph.replay()
        actual = self._device_output.cpu().numpy()
        difference = float(np.abs(actual - expected).max())
        if not np.isfinite(actual).all() or difference > self.MAX_ABS_DIFFERENCE:
            raise RuntimeError(
                f"CUDA graph output differs from the eager path (max abs difference {difference:.3e})"
            )
        logging.info("CUDA graph matches the eager path: max abs difference %.2e", difference)

    def run(self, state: np.ndarray, images: dict[str, np.ndarray], marks: list[tuple[str, float]]) -> np.ndarray:
        runtime = self._runtime
        with self._torch.inference_mode():
            self._upload(state, images)
            self._noise.normal_()
            runtime._profile_mark(marks, "upload")
            self._graph.replay()
            runtime._profile_mark(marks, "replay")
            # The host copy waits for the replay; `_upload` may then reuse the
            # pinned buffers on the next call because everything before it is done.
            chunk = self._device_output.cpu().numpy()
        runtime._profile_mark(marks, "cpu")
        return chunk
