#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.adapters.teleavatar import (
    ACTION,
    OBS_IMAGE,
    OBS_IMAGE_2,
    OBS_IMAGE_3,
    OBS_STATE,
    TELEAVATAR_ACTION_INDICES,
    TELEAVATAR_ACTION_NAMES,
    TELEAVATAR_RAW_FEATURE_NAMES,
    TELEAVATAR_STATE_INDICES,
    TELEAVATAR_STATE_NAMES,
    TeleavatarV2Adapter,
    adapt_teleavatar_action,
    crop_left_stereo_eye,
    get_dataset_adapter,
    gripper_effort_to_trigger,
)


def make_stats(size: int, offset: float = 0.0) -> dict:
    values = np.arange(size, dtype=np.float32) + offset
    return {
        "mean": values.copy(),
        "std": np.ones(size, dtype=np.float32),
        "min": values.copy() - 1,
        "max": values.copy() + 1,
        "count": np.array([10]),
    }


def serialize_stats(stats: dict) -> dict:
    return {
        feature: {name: value.tolist() for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def make_adapted_stats() -> dict:
    episode_zero = {OBS_STATE: make_stats(14), ACTION: make_stats(16, 100)}
    episode_one = {OBS_STATE: make_stats(14, 10), ACTION: make_stats(16, 200)}
    global_stats = {OBS_STATE: make_stats(14, 5), ACTION: make_stats(16, 150)}
    return {
        "version": 1,
        "camera_eye": "left",
        "state_indices": list(TELEAVATAR_STATE_INDICES),
        "action_indices": list(TELEAVATAR_ACTION_INDICES),
        "image_key_mapping": {
            "observation.images.head_camera": OBS_IMAGE,
            "observation.images.left_color": OBS_IMAGE_2,
            "observation.images.right_color": OBS_IMAGE_3,
        },
        "stats": serialize_stats(global_stats),
        "episodes": {
            "0": serialize_stats(episode_zero),
            "1": serialize_stats(episode_one),
        },
    }


def make_image_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channels"],
    }


class TeleavatarV2AdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = TeleavatarV2Adapter(adapted_stats=make_adapted_stats())
        self.metadata = SimpleNamespace(
            info={
                "features": {
                    OBS_STATE: {
                        "dtype": "float32",
                        "shape": (72,),
                        "names": list(TELEAVATAR_RAW_FEATURE_NAMES),
                    },
                    ACTION: {
                        "dtype": "float32",
                        "shape": (72,),
                        "names": list(TELEAVATAR_RAW_FEATURE_NAMES),
                    },
                    "observation.images.head_camera": make_image_feature(1920, 3840),
                    "observation.images.left_color": make_image_feature(800, 2560),
                    "observation.images.right_color": make_image_feature(800, 2560),
                }
            },
            stats={
                OBS_STATE: make_stats(72),
                ACTION: make_stats(72),
                "observation.images.head_camera": make_stats(3),
            },
            episodes_stats={
                0: {OBS_STATE: make_stats(72), ACTION: make_stats(72)},
                1: {OBS_STATE: make_stats(72), ACTION: make_stats(72)},
            },
        )

    def test_adapts_metadata_schema_stats_and_camera_order(self):
        self.adapter.adapt_metadata(self.metadata)

        features = self.metadata.info["features"]
        self.assertEqual(features[OBS_STATE]["shape"], (14,))
        self.assertEqual(tuple(features[OBS_STATE]["names"]), TELEAVATAR_STATE_NAMES)
        self.assertEqual(features[ACTION]["shape"], (16,))
        self.assertEqual(tuple(features[ACTION]["names"]), TELEAVATAR_ACTION_NAMES)
        self.assertEqual(features[OBS_IMAGE]["shape"], (1920, 1920, 3))
        self.assertEqual(features[OBS_IMAGE_2]["shape"], (800, 1280, 3))
        self.assertEqual(features[OBS_IMAGE_3]["shape"], (800, 1280, 3))
        self.assertEqual(
            [key for key in features if key in (OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3)],
            [OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3],
        )

        np.testing.assert_array_equal(
            self.metadata.stats[OBS_STATE]["mean"], np.arange(14) + 5
        )
        np.testing.assert_array_equal(
            self.metadata.stats[ACTION]["mean"], np.arange(16) + 150
        )
        self.assertIn(OBS_IMAGE, self.metadata.stats)
        self.assertNotIn("observation.images.head_camera", self.metadata.stats)
        np.testing.assert_array_equal(
            self.metadata.episodes_stats[0][ACTION]["mean"], np.arange(16) + 100
        )
        np.testing.assert_array_equal(
            self.metadata.episodes_stats[1][ACTION]["mean"], np.arange(16) + 200
        )

    def test_adapts_state_action_and_all_three_left_eyes(self):
        raw_state = np.arange(72, dtype=np.float32)
        raw_action = np.zeros((2, 72), dtype=np.float32)
        raw_action[:, :7] = np.arange(7)
        raw_action[:, 8:15] = np.arange(10, 17)
        raw_action[:, 39] = [2.0, 0.0]
        raw_action[:, 47] = [-1.6, 2.0]

        head = torch.arange(3 * 8 * 16).reshape(3, 8, 16)
        left = torch.arange(3 * 4 * 16).reshape(3, 4, 16)
        right = torch.arange(3 * 4 * 16).reshape(3, 4, 16)
        item = {
            OBS_STATE: raw_state,
            ACTION: raw_action,
            "observation.images.head_camera": head,
            "observation.images.left_color": left,
            "observation.images.right_color": right,
        }

        adapted = self.adapter.adapt_item(item)

        np.testing.assert_array_equal(
            adapted[OBS_STATE], raw_state[list(TELEAVATAR_STATE_INDICES)]
        )
        self.assertEqual(adapted[ACTION].shape, (2, 16))
        np.testing.assert_allclose(adapted[ACTION][:, 7], [0.0, 0.1], atol=1e-6)
        np.testing.assert_allclose(adapted[ACTION][:, 15], [1.0, 0.0], atol=1e-6)
        np.testing.assert_array_equal(
            adapted[ACTION][0, :7], raw_action[0, list(TELEAVATAR_ACTION_INDICES[:7])]
        )
        self.assertTrue(torch.equal(adapted[OBS_IMAGE], head[..., :8]))
        self.assertTrue(torch.equal(adapted[OBS_IMAGE_2], left[..., :8]))
        self.assertTrue(torch.equal(adapted[OBS_IMAGE_3], right[..., :8]))

    def test_effort_conversion_boundaries_for_numpy_and_torch(self):
        effort = np.array([2.0, 0.0, -1.6], dtype=np.float32)
        expected = np.array([0.0, 0.1, 1.0], dtype=np.float32)
        np.testing.assert_allclose(
            gripper_effort_to_trigger(effort), expected, atol=1e-6
        )
        torch.testing.assert_close(
            gripper_effort_to_trigger(torch.from_numpy(effort)),
            torch.from_numpy(expected),
        )

    def test_action_uses_effort_indices_not_gripper_positions(self):
        raw_action = np.zeros(72, dtype=np.float32)
        raw_action[7] = 999.0
        raw_action[15] = 999.0
        raw_action[39] = 0.0
        raw_action[47] = -1.6
        adapted = adapt_teleavatar_action(raw_action)
        np.testing.assert_allclose(adapted[[7, 15]], [0.1, 1.0], atol=1e-6)

    def test_crop_left_eye_supports_hwc_and_skips_single_eye(self):
        stereo_hwc = np.arange(8 * 16 * 3).reshape(8, 16, 3)
        np.testing.assert_array_equal(
            crop_left_stereo_eye(stereo_hwc), stereo_hwc[:, :8]
        )

        single_eye_chw = torch.zeros(3, 8, 12)
        self.assertIs(crop_left_stereo_eye(single_eye_chw), single_eye_chw)

    def test_missing_adapted_stats_fails_loudly(self):
        adapter = TeleavatarV2Adapter()
        with self.assertRaisesRegex(
            FileNotFoundError, "compute_teleavatar_v2_stats.py"
        ):
            adapter.adapt_metadata(self.metadata)

    def test_ignores_other_robot_types(self):
        self.assertIsNone(get_dataset_adapter("so100"))


if __name__ == "__main__":
    unittest.main()
