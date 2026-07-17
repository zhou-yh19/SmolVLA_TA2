#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

import numpy as np

from lerobot.datasets.adapters.teleavatar import (
    TELEAVATAR_JOINT_POSITION_NAMES,
    get_dataset_adapter,
)

ACTION = "action"
OBS_STATE = "observation.state"


def make_stats(size: int) -> dict:
    values = np.arange(size, dtype=np.float32)
    return {
        "mean": values.copy(),
        "std": values.copy() + 1,
        "min": values.copy() - 1,
        "max": values.copy() + 2,
        "count": np.array([10]),
    }


class TeleavatarJointPositionAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = get_dataset_adapter("teleavatar")
        feature_names = list(TELEAVATAR_JOINT_POSITION_NAMES) + [f"extra_{index}" for index in range(56)]
        self.metadata = SimpleNamespace(
            info={
                "features": {
                    OBS_STATE: {"dtype": "float32", "shape": (72,), "names": feature_names.copy()},
                    ACTION: {"dtype": "float32", "shape": (72,), "names": feature_names.copy()},
                }
            },
            stats={OBS_STATE: make_stats(72), ACTION: make_stats(72)},
            episodes_stats={0: {OBS_STATE: make_stats(72), ACTION: make_stats(72)}},
        )

    def test_projects_metadata_and_statistics(self):
        self.adapter.adapt_metadata(self.metadata)

        for key in (OBS_STATE, ACTION):
            self.assertEqual(self.metadata.info["features"][key]["shape"], (16,))
            self.assertEqual(
                tuple(self.metadata.info["features"][key]["names"]), TELEAVATAR_JOINT_POSITION_NAMES
            )
            self.assertEqual(self.metadata.stats[key]["mean"].shape, (16,))
            self.assertEqual(self.metadata.episodes_stats[0][key]["mean"].shape, (16,))
            self.assertEqual(self.metadata.stats[key]["count"].shape, (1,))

    def test_projects_sample_last_dimension(self):
        item = {
            OBS_STATE: np.arange(72, dtype=np.float32).reshape(1, 72),
            ACTION: np.arange(50 * 72, dtype=np.float32).reshape(50, 72),
            "task": "move both arms",
        }

        projected = self.adapter.adapt_item(item)

        self.assertEqual(projected[OBS_STATE].shape, (1, 16))
        self.assertEqual(projected[ACTION].shape, (50, 16))
        np.testing.assert_array_equal(projected[OBS_STATE], item[OBS_STATE][..., :16])
        np.testing.assert_array_equal(projected[ACTION], item[ACTION][..., :16])

    def test_ignores_other_robot_types(self):
        self.assertIsNone(get_dataset_adapter("so100"))


if __name__ == "__main__":
    unittest.main()
