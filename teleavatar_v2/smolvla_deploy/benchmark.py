"""Offline inference benchmark: real checkpoint, synthetic observation, no robot.

The `dry` stage still needs live sensors, because it exercises the real
observation path. This one deliberately does not: it feeds the policy random
frames of the right shape so inference latency can be measured on a laptop or a
robot host with nothing plugged in. Only the timings mean anything here - the
actions come from noise.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np

from .contracts import ACTION_DIM, OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3, OBS_STATE, STATE_DIM
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
    parser.add_argument(
        "--sampler",
        choices=("euler", "streamtp", "compare"),
        default="euler",
        help="Original Euler, accelerated StreamTP, or matched-noise A/B comparison",
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Override flow denoising steps")
    parser.add_argument("--streamtp-tolerance", type=float, default=0.02)
    parser.add_argument("--streamtp-max-sweeps", type=int, default=None)
    parser.add_argument("--streamtp-anderson-depth", type=int, default=3)
    parser.add_argument("--streamtp-anderson-regularization", type=float, default=1e-4)
    parser.add_argument("--streamtp-no-warm-start", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="Time the eager kernel-by-kernel path instead of the CUDA graph replay",
    )
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
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        help="Append per-call stage timings, StreamTP diagnostics, and matched-noise RMS to JSONL",
    )
    parser.add_argument("--summary-json", type=Path, help="Write aggregate benchmark results as JSON")
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

    runtime_sampler = "streamtp" if args.sampler == "compare" else args.sampler
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
        cuda_graph=args.cuda_graph,
        sampler=runtime_sampler,
        streamtp_tolerance=args.streamtp_tolerance,
        streamtp_max_sweeps=args.streamtp_max_sweeps,
        streamtp_anderson_depth=args.streamtp_anderson_depth,
        streamtp_anderson_regularization=args.streamtp_anderson_regularization,
        streamtp_warm_start=not args.streamtp_no_warm_start,
        streamtp_shift_steps=args.execution_horizon,
        collect_metrics=True,
    )

    observation = _synthetic_observation(runtime, args.image_hw, args.seed)
    logging.info(
        "Synthetic cameras: %s",
        ", ".join(f"{key.rsplit('.', 1)[-1]}={observation[key].shape}" for key in CAMERAS),
    )

    generator = runtime._torch.Generator(device=runtime._device).manual_seed(args.seed)
    for index in range(args.warmup):
        if args.sampler == "compare":
            noise = runtime.sample_noise(generator)
            runtime.infer_action_chunk(
                observation, args.task, sampler="euler", noise=noise, record_metrics=False
            )
            runtime.infer_action_chunk(
                observation, args.task, sampler="streamtp", noise=noise, record_metrics=False
            )
        else:
            runtime.infer_action_chunk(observation, args.task, record_metrics=False)
        if index == 0 and (args.compile_model or runtime._cuda_graph_enabled):
            logging.info("First iteration done; compilation/capture is out of the way")

    records: list[dict[str, object]] = []
    timing_groups: dict[str, list[float]] = {"euler": [], "streamtp": []}
    errors: list[float] = []
    if args.sampler == "compare":
        for index in range(args.iters):
            noise = runtime.sample_noise(generator)
            order = ("euler", "streamtp") if index % 2 == 0 else ("streamtp", "euler")
            outputs = {}
            pair_records = {}
            for sampler in order:
                if runtime._device.type == "cuda":
                    runtime._torch.cuda.synchronize(runtime._device)
                    memory_baseline = runtime._torch.cuda.memory_allocated(runtime._device)
                    runtime._torch.cuda.reset_peak_memory_stats(runtime._device)
                _, latency = runtime.infer_action_chunk(
                    observation, args.task, sampler=sampler, noise=noise
                )
                timing_groups[sampler].append(latency)
                outputs[sampler] = runtime.last_normalized_actions[:, :ACTION_DIM].copy()
                pair_records[sampler] = dict(runtime.last_metrics)
                if runtime._device.type == "cuda":
                    memory_peak = runtime._torch.cuda.max_memory_allocated(runtime._device)
                    pair_records[sampler]["incremental_peak_memory_mib"] = max(
                        0.0, (memory_peak - memory_baseline) / (1024**2)
                    )
            rms = float(np.sqrt(np.mean(np.square(outputs["streamtp"] - outputs["euler"]))))
            errors.append(rms)
            for sampler in ("euler", "streamtp"):
                pair_records[sampler]["pair"] = index + 1
                pair_records[sampler]["matched_noise_euler_rms"] = rms
                records.append(pair_records[sampler])
    else:
        for _ in range(args.iters):
            if runtime._device.type == "cuda":
                runtime._torch.cuda.synchronize(runtime._device)
                memory_baseline = runtime._torch.cuda.memory_allocated(runtime._device)
                runtime._torch.cuda.reset_peak_memory_stats(runtime._device)
            _, latency = runtime.infer_action_chunk(observation, args.task)
            timing_groups[args.sampler].append(latency)
            record = dict(runtime.last_metrics)
            if runtime._device.type == "cuda":
                memory_peak = runtime._torch.cuda.max_memory_allocated(runtime._device)
                record["incremental_peak_memory_mib"] = max(
                    0.0, (memory_peak - memory_baseline) / (1024**2)
                )
            records.append(record)

    def summarize(timings: list[float]) -> dict[str, float]:
        ordered = sorted(timings)
        p90_index = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        return {
            "median_ms": statistics.median(ordered),
            "p90_ms": ordered[p90_index],
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        }

    summary = {
        sampler: summarize(timings)
        for sampler, timings in timing_groups.items()
        if timings
    }
    summary["config"] = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "device": str(runtime._device),
        "gpu": (
            runtime._torch.cuda.get_device_name(runtime._device)
            if runtime._device.type == "cuda"
            else None
        ),
        "precision_override": args.precision,
        "attention_implementation": args.attn_implementation,
        "num_steps": int(runtime.policy.config.num_steps),
        "iterations": args.iters,
        "warmup_iterations": args.warmup,
        "streamtp_tolerance": args.streamtp_tolerance,
        "streamtp_max_sweeps": runtime._streamtp_config.max_sweeps,
        "streamtp_anderson_depth": args.streamtp_anderson_depth,
        "streamtp_warm_start": not args.streamtp_no_warm_start,
        "streamtp_shift_steps": args.execution_horizon,
        "seed": args.seed,
    }
    for sampler in ("euler", "streamtp"):
        if sampler not in summary:
            continue
        values = summary[sampler]
        logging.info(
            "%s over %d iters: median=%.1fms p90=%.1fms min=%.1fms max=%.1fms",
            sampler,
            len(timing_groups[sampler]),
            values["median_ms"],
            values["p90_ms"],
            values["min_ms"],
            values["max_ms"],
        )
    if errors:
        summary["comparison"] = {
            "mean_matched_noise_euler_rms": statistics.mean(errors),
            "max_matched_noise_euler_rms": max(errors),
            "median_speedup": summary["euler"]["median_ms"] / summary["streamtp"]["median_ms"],
        }
        logging.info(
            "matched-noise comparison: speedup=%.3fx mean_rms=%.6f max_rms=%.6f",
            summary["comparison"]["median_speedup"],
            summary["comparison"]["mean_matched_noise_euler_rms"],
            summary["comparison"]["max_matched_noise_euler_rms"],
        )

    streamtp_records = [record["streamtp"] for record in records if "streamtp" in record]
    summary["component_metrics"] = {}
    for sampler in ("euler", "streamtp"):
        module_records = [record[sampler] for record in records if sampler in record]
        memory_records = [
            record["incremental_peak_memory_mib"]
            for record in records
            if record["sampler"] == sampler and "incremental_peak_memory_mib" in record
        ]
        if module_records or memory_records:
            summary["component_metrics"][sampler] = {}
        if module_records:
            summary["component_metrics"][sampler].update(
                median_prefix_ms=statistics.median(item["prefix_ms"] for item in module_records),
                median_solver_ms=statistics.median(item["solver_ms"] for item in module_records),
            )
        if memory_records:
            summary["component_metrics"][sampler]["median_incremental_peak_memory_mib"] = (
                statistics.median(memory_records)
            )
    if streamtp_records:
        summary["streamtp_diagnostics"] = {
            "mean_sweeps": statistics.mean(item["sweeps"] for item in streamtp_records),
            "mean_expert_evaluations": statistics.mean(
                item["expert_evaluations"] for item in streamtp_records
            ),
            "fallback_count": sum(bool(item["fallback"]) for item in streamtp_records),
            "fallback_rate": statistics.mean(bool(item["fallback"]) for item in streamtp_records),
            "mean_final_residual_rms": statistics.mean(
                item["final_residual_rms"] for item in streamtp_records
            ),
            "median_prefix_ms": statistics.median(item["prefix_ms"] for item in streamtp_records),
            "median_solver_ms": statistics.median(item["solver_ms"] for item in streamtp_records),
        }
        warm_records = [item for item in streamtp_records if item["warm_plan_shift_rms"] is not None]
        if warm_records:
            summary["streamtp_diagnostics"]["mean_warm_plan_shift_rms"] = statistics.mean(
                item["warm_plan_shift_rms"] for item in warm_records
            )
        if len(warm_records) >= 2:
            shifts = np.asarray([item["warm_plan_shift_rms"] for item in warm_records])
            sweeps = np.asarray([item["sweeps"] for item in warm_records])
            if shifts.std() > 0 and sweeps.std() > 0:
                summary["streamtp_diagnostics"]["plan_shift_sweep_pearson_r"] = float(
                    np.corrcoef(shifts, sweeps)[0, 1]
                )
        logging.info(
            "StreamTP diagnostics: mean_sweeps=%.2f mean_evals=%.1f fallback=%d/%d "
            "mean_residual=%.6f prefix_median=%.1fms solver_median=%.1fms",
            summary["streamtp_diagnostics"]["mean_sweeps"],
            summary["streamtp_diagnostics"]["mean_expert_evaluations"],
            summary["streamtp_diagnostics"]["fallback_count"],
            len(streamtp_records),
            summary["streamtp_diagnostics"]["mean_final_residual_rms"],
            summary["streamtp_diagnostics"]["median_prefix_ms"],
            summary["streamtp_diagnostics"]["median_solver_ms"],
        )

    if args.metrics_jsonl is not None:
        args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_jsonl.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        logging.info("Wrote %d per-call records to %s", len(records), args.metrics_jsonl)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        logging.info("Wrote summary to %s", args.summary_json)

    # What the control loop actually needs: one chunk must be inferred while the
    # previous chunk is executing, or the arms stall between chunks.
    if args.control_frequency > 0 and args.execution_horizon > 0:
        budget_ms = 1000.0 * args.execution_horizon / args.control_frequency
        for sampler, timings in timing_groups.items():
            if not timings:
                continue
            values = summary[sampler]
            p90 = values["p90_ms"]
            verdict = "fits" if p90 < budget_ms else "OVER BUDGET"
            logging.info(
                "%s chunk budget at %.0fHz x %d actions = %.0fms; p90 uses %.0f%% of it (%s)",
                sampler,
                args.control_frequency,
                args.execution_horizon,
                budget_ms,
                100.0 * p90 / budget_ms,
                verdict,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
