"""ROS2 and RTP hardware interface for direct-trigger SmolVLA deployment."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from threading import Lock
from typing import Optional

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from .contracts import build_state, gripper_position, map_smolvla_images, split_trigger_action
from .rtp_video_interface import RTPH265VideoInterface


class TeleavatarSmolVLAInterface(Node):
    """Thread-safe TeleAvatar V2 interface with SmolVLA-native action semantics."""

    LEFT_NAMES = [f"l_joint{i}" for i in range(1, 8)]
    RIGHT_NAMES = [f"r_joint{i}" for i in range(1, 8)]

    def __init__(
        self,
        *,
        arm_config: str | Path,
        node_name: str = "teleavatar_smolvla_interface",
        rtp_port: int = 8890,
        rtp_payload: int = 96,
        decoder: str = "nvh265dec max-display-delay=0",
        sensor_timeout: float = 1.0,
        max_joint_step_rad: float = 0.0,
    ) -> None:
        super().__init__(node_name)
        self._lock = Lock()
        self._sensor_timeout = float(sensor_timeout)
        self._max_joint_step_rad = float(max_joint_step_rad)
        self._joint_states: dict[str, JointState] = {}
        self._joint_timestamps: dict[str, float] = {}
        self._last_arm_command: dict[str, np.ndarray] = {}

        config_path = Path(arm_config).expanduser().resolve()
        config = yaml.safe_load(config_path.read_text())
        self._joint_lower = {
            "left_arm": np.asarray(config["arms"]["left_arm"]["lower"], dtype=np.float64),
            "right_arm": np.asarray(config["arms"]["right_arm"]["lower"], dtype=np.float64),
        }
        self._joint_upper = {
            "left_arm": np.asarray(config["arms"]["left_arm"]["upper"], dtype=np.float64),
            "right_arm": np.asarray(config["arms"]["right_arm"]["upper"], dtype=np.float64),
        }

        self._video = RTPH265VideoInterface(port=rtp_port, payload=rtp_payload, decoder=decoder)
        self._video.start()

        self.create_subscription(
            JointState,
            "/left_arm/joint_states",
            lambda msg: self._on_joint_state("left_arm", msg),
            10,
        )
        self.create_subscription(
            JointState,
            "/right_arm/joint_states",
            lambda msg: self._on_joint_state("right_arm", msg),
            10,
        )
        self.create_subscription(
            JointState,
            "/left_gripper/joint_states",
            lambda msg: self._on_joint_state("left_gripper", msg),
            10,
        )
        self.create_subscription(
            JointState,
            "/right_gripper/joint_states",
            lambda msg: self._on_joint_state("right_gripper", msg),
            10,
        )
        self._arm_publishers = {
            "left_arm": self.create_publisher(JointState, "/api/left_arm/joint_cmd", 10),
            "right_arm": self.create_publisher(JointState, "/api/right_arm/joint_cmd", 10),
        }
        self._gripper_publishers = {
            "left": self.create_publisher(Float32, "/api/left_gripper/cmd", 10),
            "right": self.create_publisher(Float32, "/api/right_gripper/cmd", 10),
        }
        self._enable_publisher = self.create_publisher(Float32, "/api/fsm/enable", 10)
        self.get_logger().info("TeleAvatar SmolVLA interface initialized")

    def _on_joint_state(self, arm: str, msg: JointState) -> None:
        with self._lock:
            self._joint_states[arm] = msg
            self._joint_timestamps[arm] = time.time()

    @staticmethod
    def _ordered_positions(msg: JointState, expected_names: list[str]) -> np.ndarray:
        if msg.name:
            positions = dict(zip(msg.name, msg.position, strict=False))
            missing = [name for name in expected_names if name not in positions]
            if missing:
                raise ValueError(f"JointState is missing joints: {missing}")
            return np.asarray([positions[name] for name in expected_names], dtype=np.float32)
        if len(msg.position) < 7:
            raise ValueError(f"JointState has only {len(msg.position)} positions")
        return np.asarray(msg.position[:7], dtype=np.float32)

    def _sensor_failures(self, *, include_images: bool) -> list[str]:
        now = time.time()
        failures: list[str] = []
        if self._video.stream_ended():
            failures.append("RTP pipeline stopped")

        with self._lock:
            stamps = dict(self._joint_timestamps)
        for arm in ("left_arm", "right_arm", "left_gripper", "right_gripper"):
            stamp = stamps.get(arm)
            if stamp is None:
                failures.append(f"{arm} not received")
            elif now - stamp > self._sensor_timeout:
                failures.append(f"{arm} stale by {now - stamp:.2f}s")

        if include_images:
            image_stamps = self._video.get_image_timestamps()
            for view in ("head_left_eye", "left_wrist_left_eye", "right_wrist_left_eye"):
                stamp = image_stamps.get(view)
                if stamp is None:
                    failures.append(f"{view} not received")
                elif now - stamp > self._sensor_timeout:
                    failures.append(f"{view} stale by {now - stamp:.2f}s")
        return failures

    def wait_for_initial_data(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._sensor_failures(include_images=True):
                self.get_logger().info("All joint and camera inputs are ready")
                return True
            time.sleep(0.1)
        self.get_logger().error(
            "Timed out waiting for sensors: " + "; ".join(self._sensor_failures(include_images=True))
        )
        return False

    def sensors_healthy(self) -> bool:
        failures = self._sensor_failures(include_images=True)
        if failures:
            self.get_logger().error("Sensor check failed: " + "; ".join(failures))
            return False
        return True

    def get_observation(self) -> Optional[dict]:
        """Return one SmolVLA observation, or None if any sensor is stale."""
        failures = self._sensor_failures(include_images=True)
        if failures:
            self.get_logger().error("Observation unavailable: " + "; ".join(failures))
            return None

        split_images, _ = self._video.get_latest_images_with_timestamps()
        with self._lock:
            left_msg = self._joint_states["left_arm"]
            right_msg = self._joint_states["right_arm"]
            left_gripper_msg = self._joint_states["left_gripper"]
            right_gripper_msg = self._joint_states["right_gripper"]
        try:
            left = self._ordered_positions(left_msg, self.LEFT_NAMES)
            right = self._ordered_positions(right_msg, self.RIGHT_NAMES)
            return {
                "observation.state": build_state(
                    left, right,
                    gripper_position(left_gripper_msg.position),
                    gripper_position(right_gripper_msg.position),
                ),
                **map_smolvla_images(split_images),
            }
        except (KeyError, ValueError) as exc:
            self.get_logger().error(f"Invalid sensor schema: {exc}")
            return None

    def _bounded_arm_command(self, arm: str, target: np.ndarray) -> np.ndarray:
        bounded = np.clip(target, self._joint_lower[arm], self._joint_upper[arm])
        if self._max_joint_step_rad > 0:
            reference = self._last_arm_command.get(arm)
            if reference is None:
                with self._lock:
                    msg = self._joint_states.get(arm)
                if msg is not None:
                    names = self.LEFT_NAMES if arm == "left_arm" else self.RIGHT_NAMES
                    reference = self._ordered_positions(msg, names)
            if reference is not None:
                bounded = np.clip(
                    bounded,
                    reference - self._max_joint_step_rad,
                    reference + self._max_joint_step_rad,
                )
        self._last_arm_command[arm] = bounded.copy()
        return bounded

    def publish_trigger_action(self, action: np.ndarray) -> None:
        """Publish a SmolVLA action; dimensions 7 and 15 are direct triggers."""
        command = split_trigger_action(action)
        stamp = self.get_clock().now().to_msg()

        enable = Float32()
        enable.data = 1.0
        self._enable_publisher.publish(enable)

        for arm, names, target in (
            ("left_arm", self.LEFT_NAMES, command.left_arm),
            ("right_arm", self.RIGHT_NAMES, command.right_arm),
        ):
            msg = JointState()
            msg.header.stamp = stamp
            msg.header.frame_id = arm
            msg.name = names
            msg.position = self._bounded_arm_command(arm, target).tolist()
            msg.velocity = [0.0] * 7
            msg.effort = [0.0] * 7
            self._arm_publishers[arm].publish(msg)

        left_gripper = Float32()
        left_gripper.data = command.left_trigger
        self._gripper_publishers["left"].publish(left_gripper)
        right_gripper = Float32()
        right_gripper.data = command.right_trigger
        self._gripper_publishers["right"].publish(right_gripper)

    def destroy_node(self) -> None:
        try:
            self._video.stop()
        finally:
            super().destroy_node()
