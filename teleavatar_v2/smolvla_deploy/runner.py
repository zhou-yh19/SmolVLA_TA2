"""Single-process SmolVLA inference and TeleAvatar V2 control loop."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .contracts import select_execution_actions
from .policy_runtime import SmolVLARuntime
from .robot_interface import TeleavatarSmolVLAInterface


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
    parser.add_argument("--task", required=True, help="Language instruction used during inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override SmolVLA flow denoising steps from the checkpoint config",
    )
    parser.add_argument(
        "--profile-inference",
        action="store_true",
        help="Log per-stage inference timing. Adds CUDA synchronizations, so use for diagnosis only.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default=None,
        help="Backbone dtype; defaults to bf16 on CUDA and fp32 on CPU. "
        "The checkpoint is stored in fp32, so fp32 runs the SigLIP tower and both "
        "transformer stacks without tensor cores.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default="sdpa",
        help="Attention kernel. 'eager' reproduces training exactly; 'sdpa' is the deployment default.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="torch.compile the denoise step. Costs compilation time on the first chunk.",
    )
    parser.add_argument("--control-frequency", type=float, default=20.0)
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--sensor-timeout", type=float, default=1.0)
    parser.add_argument("--rtp-port", type=int, default=8890)
    parser.add_argument("--rtp-payload", type=int, default=96)
    parser.add_argument("--decoder", default="nvh265dec max-display-delay=0")
    parser.add_argument("--arm-config", type=Path, default=project_root / "arm_config.yml")
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.0,
        help="Optional per-command joint-step limiter; 0 disables it",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually publish robot commands. Without this flag, run one inference chunk and exit.",
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="Stop after N chunks; 0 means unlimited")
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=2,
        help="Untimed inferences before the control loop, on real observations "
        "whose actions are discarded. With --compile the first one pays "
        "compilation, which would otherwise land on chunk 1.",
    )
    return parser


def _wait_for_tick(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    return deadline


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.control_frequency <= 0:
        raise ValueError("control-frequency must be positive")
    if args.execution_horizon <= 0:
        raise ValueError("execution-horizon must be positive")
    if args.max_joint_step_rad < 0:
        raise ValueError("max-joint-step-rad cannot be negative")
    if args.warmup_chunks < 0:
        raise ValueError("warmup-chunks cannot be negative")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if not args.execute:
        logging.warning("DRY RUN: actions will be inferred but not published")

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

    rclpy.init()
    robot = TeleavatarSmolVLAInterface(
        arm_config=args.arm_config,
        rtp_port=args.rtp_port,
        rtp_payload=args.rtp_payload,
        decoder=args.decoder,
        sensor_timeout=args.sensor_timeout,
        max_joint_step_rad=args.max_joint_step_rad,
    )
    # Single-threaded on purpose. `MultiThreadedExecutor()` defaults to
    # `multiprocessing.cpu_count()` threads, and every one of them competes for
    # the GIL with the inference thread, which spends its time launching many
    # small kernels from Python. Measured on a 24-thread host with the cameras
    # streaming: 342ms median / 961ms worst chunk multi-threaded against 274ms /
    # 307ms single-threaded, for four subscriptions whose callbacks only take a
    # lock and store the message.
    executor = SingleThreadedExecutor()
    executor.add_node(robot)
    spin_thread = threading.Thread(target=executor.spin, name="ros2-executor", daemon=True)
    spin_thread.start()

    chunk_count = 0
    try:
        if not robot.wait_for_initial_data(timeout=30.0):
            return 2

        # CUDA's first-call setup and, under --compile, dynamo's compilation both
        # land on whichever inference runs first. Spend them here: on chunk 1 a
        # 30s+ compile would leave the arms holding the previous command well
        # past the chunk budget. Nothing is published from these.
        for index in range(args.warmup_chunks):
            observation = robot.get_observation()
            if observation is None:
                raise RuntimeError("Cannot warm up from missing or stale observation")
            _, warmup_ms = runtime.infer_action_chunk(observation, args.task)
            logging.info(
                "warmup %d/%d: inference=%.1fms", index + 1, args.warmup_chunks, warmup_ms
            )

        while rclpy.ok():
            observation = robot.get_observation()
            if observation is None:
                raise RuntimeError("Cannot infer from missing or stale observation")

            chunk, inference_ms = runtime.infer_action_chunk(observation, args.task)
            actions = select_execution_actions(chunk, args.execution_horizon)
            chunk_count += 1
            logging.info(
                "chunk=%d inference=%.1fms predicted=%d executing=%d first_triggers=(%.3f, %.3f)",
                chunk_count,
                inference_ms,
                len(chunk),
                len(actions),
                actions[0, 7],
                actions[0, 15],
            )

            if not args.execute:
                return 0

            period = 1.0 / args.control_frequency
            deadline = time.monotonic()
            for action in actions:
                if not robot.sensors_healthy():
                    raise RuntimeError("Sensor became stale while executing an action chunk")
                robot.publish_trigger_action(action)
                deadline += period
                _wait_for_tick(deadline)

            if args.max_chunks and chunk_count >= args.max_chunks:
                break
    except KeyboardInterrupt:
        logging.info("Stopped by user")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        robot.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
