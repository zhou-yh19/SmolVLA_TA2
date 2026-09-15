#!/usr/bin/env python3
"""Attribute the gap between `bench` and `run` inference latency.

Runs the same checkpoint under increasingly production-like conditions and
reports per-condition latency, without publishing anything to the robot:

  idle      back-to-back inferences (what `bench` measures)
  gap       a control-loop-sized sleep between inferences (GPU/CPU clocks idle down)
  video     gap + the RTP/H.265 decoder running on the live stream
  ros       gap + an executor thread spinning the four joint-state subscriptions
            (the pre-polling design; shows how much a GIL-holding thread costs)
  all       gap + video + polled joint states, observations taken exactly as
            `run` takes them

`video`, `ros` and `all` need the robot's stream and topics to be live.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import statistics
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from smolvla_deploy.benchmark import _synthetic_observation  # noqa: E402
from smolvla_deploy.contracts import OBS_STATE, map_smolvla_images  # noqa: E402
from smolvla_deploy.policy_runtime import SmolVLARuntime  # noqa: E402

CONDITIONS = ("idle", "gap", "video", "ros", "all")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smolvla-repo", type=Path, required=True)
    parser.add_argument("--vlm-model-path", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sampler", choices=("euler", "streamtp"), default="euler")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--streamtp-tolerance", type=float, default=0.02)
    parser.add_argument("--streamtp-max-sweeps", type=int, default=None)
    parser.add_argument("--streamtp-anderson-depth", type=int, default=3)
    parser.add_argument("--streamtp-anderson-regularization", type=float, default=1e-4)
    parser.add_argument("--streamtp-no-warm-start", action="store_true")
    parser.add_argument("--streamtp-shift-steps", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-cuda-graph", dest="cuda_graph", action="store_false")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--gap-s", type=float, default=16 / 30, help="Sleep between inferences")
    parser.add_argument("--rtp-port", type=int, default=8890)
    parser.add_argument("--arm-config", type=Path, default=PROJECT_ROOT / "arm_config.yml")
    return parser


def _summarize(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    logging.info(
        "%-6s n=%d median=%.1fms p90=%.1fms min=%.1fms max=%.1fms  first->last: %s",
        name,
        len(ordered),
        statistics.median(ordered),
        p90,
        ordered[0],
        ordered[-1],
        " ".join(f"{t:.0f}" for t in timings),
    )


def _measure(runtime, get_observation, task: str, iters: int, gap_s: float) -> list[float]:
    # Two untimed inferences so each condition starts from the same warm state.
    for _ in range(2):
        runtime.infer_action_chunk(get_observation(), task)
        if gap_s > 0:
            time.sleep(gap_s)
    timings = []
    for _ in range(iters):
        if gap_s > 0:
            time.sleep(gap_s)
        timings.append(runtime.infer_action_chunk(get_observation(), task)[1])
    return timings


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S", force=True
    )
    logging.getLogger("RTPH265VideoInterface").setLevel(logging.WARNING)

    runtime = SmolVLARuntime(
        checkpoint=args.checkpoint,
        smolvla_repo=args.smolvla_repo,
        device=args.device,
        num_steps=args.num_steps,
        vlm_model_path=args.vlm_model_path,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
        compile_model=args.compile_model,
        cuda_graph=args.cuda_graph,
        sampler=args.sampler,
        streamtp_tolerance=args.streamtp_tolerance,
        streamtp_max_sweeps=args.streamtp_max_sweeps,
        streamtp_anderson_depth=args.streamtp_anderson_depth,
        streamtp_anderson_regularization=args.streamtp_anderson_regularization,
        streamtp_warm_start=not args.streamtp_no_warm_start,
        streamtp_shift_steps=args.streamtp_shift_steps,
    )
    synthetic = _synthetic_observation(runtime, None, 0)
    results: dict[str, list[float]] = {}

    if "idle" in args.conditions:
        results["idle"] = _measure(runtime, lambda: synthetic, args.task, args.iters, 0.0)
        _summarize("idle", results["idle"])
    if "gap" in args.conditions:
        results["gap"] = _measure(runtime, lambda: synthetic, args.task, args.iters, args.gap_s)
        _summarize("gap", results["gap"])

    if "video" in args.conditions:
        from smolvla_deploy.rtp_video_interface import RTPH265VideoInterface

        video = RTPH265VideoInterface(port=args.rtp_port)
        video.start()
        try:
            if not video.wait_for_initial_data(timeout=15.0):
                logging.error("no video; skipping 'video'")
            else:
                time.sleep(2.0)

                def observe_video():
                    images = video.get_latest_images()
                    return {OBS_STATE: synthetic[OBS_STATE], **map_smolvla_images(images)}

                results["video"] = _measure(runtime, observe_video, args.task, args.iters, args.gap_s)
                _summarize("video", results["video"])
        finally:
            video.stop()

    if "ros" in args.conditions or "all" in args.conditions:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        rclpy.init()

        if "ros" in args.conditions:
            class Subscriber(Node):
                def __init__(self):
                    super().__init__("diagnose_latency_subscriber")
                    self.count = 0
                    self._lock = threading.Lock()
                    for topic in (
                        "/left_arm/joint_states",
                        "/right_arm/joint_states",
                        "/left_gripper/joint_states",
                        "/right_gripper/joint_states",
                    ):
                        self.create_subscription(JointState, topic, self._on_msg, 10)

                def _on_msg(self, msg):
                    with self._lock:
                        self.count += 1

            node = Subscriber()
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            spin = threading.Thread(target=executor.spin, daemon=True)
            spin.start()
            time.sleep(2.0)
            before = node.count
            t0 = time.monotonic()
            results["ros"] = _measure(runtime, lambda: synthetic, args.task, args.iters, args.gap_s)
            rate = (node.count - before) / (time.monotonic() - t0)
            logging.info("ros: %.0f joint-state callbacks/s during the measurement", rate)
            _summarize("ros", results["ros"])
            executor.shutdown()
            spin.join(timeout=2.0)
            node.destroy_node()

        if "all" in args.conditions:
            from smolvla_deploy.robot_interface import TeleavatarSmolVLAInterface

            robot = TeleavatarSmolVLAInterface(arm_config=args.arm_config, rtp_port=args.rtp_port)
            try:
                if not robot.wait_for_initial_data(timeout=15.0):
                    logging.error("sensors not ready; skipping 'all'")
                else:
                    time.sleep(2.0)

                    def observe_all():
                        observation = robot.get_observation()
                        if observation is None:
                            raise RuntimeError("stale observation")
                        return observation

                    results["all"] = _measure(runtime, observe_all, args.task, args.iters, args.gap_s)
                    _summarize("all", results["all"])
            finally:
                robot.destroy_node()
        rclpy.shutdown()

    logging.info("summary (median ms): %s", "  ".join(f"{k}={statistics.median(v):.0f}" for k, v in results.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
