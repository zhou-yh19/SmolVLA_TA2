#!/usr/bin/env python3
"""Read one SmolVLA-format observation without loading or commanding a policy."""

import argparse
from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from smolvla_deploy.robot_interface import TeleavatarSmolVLAInterface  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-config", type=Path, default=PROJECT_ROOT / "arm_config.yml")
    parser.add_argument("--rtp-port", type=int, default=8890)
    args = parser.parse_args()

    rclpy.init()
    robot = TeleavatarSmolVLAInterface(arm_config=args.arm_config, rtp_port=args.rtp_port)
    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        if not robot.wait_for_initial_data(timeout=30.0):
            return 1
        observation = robot.get_observation()
        if observation is None:
            return 1
        for key, value in observation.items():
            array = np.asarray(value)
            print(f"{key}: shape={array.shape} dtype={array.dtype} range=[{array.min():.3f}, {array.max():.3f}]")
        return 0
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
