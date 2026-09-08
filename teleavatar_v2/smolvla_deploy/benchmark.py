"""Offline inference benchmark: real checkpoint, synthetic observation, no robot.

The `dry` stage still needs live sensors, because it exercises the real
observation path. This one deliberately does not: it feeds the policy random
frames of the right shape so inference latency can be measured on a laptop or a
robot host with nothing plugged in. Only the timings mean anything here - the
actions come from noise.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import statistics
import time

import numpy as np

from .contracts import OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3, OBS_STATE, STATE_DIM
from .policy_runtime import SmolVLARuntime

CAMERAS = (OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3)


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint pretrained_model directory")
    parser.add_argument("--smolvla-repo", type=Path, required=True, help="SmolVLA_TA2 repository root")
    parser.add_argument(
        "--vlm-model-path",
        type=Path,
        help="Optional local SmolVLM/processor directory for offline deployment",
    )
    parser.add_argument(
        "--task",
        default="pick up the block",
        help="Language instruction. Only its token count affects timing.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-steps", type=int, default=None, help="Override flow denoising steps")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Log the per-stage breakdown for each timed iteration",
    )
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations")
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed iterations first. CUDA needs a few; --compile needs at least one.",
    )
    parser.add_argument(
        "--image-hw",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Synthesize every camera at this size instead of the resolution the "
        "checkpoint was trained at. The policy resizes to 512x512 either way, so "
        "this only changes the host-side conversion and the PCIe transfer.",
    )
    parser.add_argument("--seed", type=int, default=0)
    # Only used to report headroom against the control loop's chunk budget.
    parser.add_argument("--control-frequency", type=float, default=20.0)
    parser.add_argument("--execution-horizon", type=int, default=16)
    return parser


def _synthetic_observation(runtime: SmolVLARuntime, image_hw, seed: int) -> dict:
    """Random frames shaped like the ones the RTP splitter would deliver.

    Timing does not depend on pixel content, but it does depend on dtype: the
    decoder hands over uint8, and `_make_batch` keeps that across PCIe.
    """
    rng = np.random.default_rng(seed)
    observation = {OBS_STATE: rng.standard_normal(STATE_DIM).astype(np.float32)}
    for key in CAMERAS:
        height, width = image_hw or runtime._trained_image_hw.get(key, (480, 640))
        observation[key] = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return observation


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.iters <= 0:
        raise ValueError("iters must be positive")
    if args.warmup < 0:
        raise ValueError("warmup cannot be negative")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.warning("BENCHMARK: random observations, no robot. Only the timings are meaningful.")

    runtime = SmolVLARuntime(
        checkpoint=args.checkpoint,
        smolvla_repo=args.smolvla_repo,
        device=args.device,
        num_steps=args.num_steps,
        profile_inference=args.profile_inference,
        vlm_model_path=args.vlm_model_path,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        compile_model=args.compile_model,
    )

    observation = _synthetic_observation(runtime, args.image_hw, args.seed)
    logging.info(
        "Synthetic cameras: %s",
        ", ".join(f"{key.rsplit('.', 1)[-1]}={observation[key].shape}" for key in CAMERAS),
    )

    for index in range(args.warmup):
        runtime.infer_action_chunk(observation, args.task)
        if index == 0 and args.compile_model:
            logging.info("First iteration done; compilation is out of the way")

    timings = [runtime.infer_action_chunk(observation, args.task)[1] for _ in range(args.iters)]
    timings.sort()

    median = statistics.median(timings)
    # Index rather than interpolate: with 20 samples an interpolated p90 invents
    # a latency that was never observed.
    p90 = timings[min(len(timings) - 1, int(round(0.9 * (len(timings) - 1))))]
    logging.info(
        "inference over %d iters: median=%.1fms p90=%.1fms min=%.1fms max=%.1fms",
        len(timings),
        median,
        p90,
        timings[0],
        timings[-1],
    )

    # What the control loop actually needs: one chunk must be inferred while the
    # previous chunk is executing, or the arms stall between chunks.
    if args.control_frequency > 0 and args.execution_horizon > 0:
        budget_ms = 1000.0 * args.execution_horizon / args.control_frequency
        verdict = "fits" if p90 < budget_ms else "OVER BUDGET"
        logging.info(
            "chunk budget at %.0fHz x %d actions = %.0fms; p90 uses %.0f%% of it (%s)",
            args.control_frequency,
            args.execution_horizon,
            budget_ms,
            100.0 * p90 / budget_ms,
            verdict,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
