#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.adapters.teleavatar import (
    ACTION,
    OBS_STATE,
    TELEAVATAR_RAW_FEATURE_NAMES,
)
from scripts.compute_teleavatar_v2_stats import compute_stats


def make_image_feature(height: int, width: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
    }


class ComputeTeleavatarV2StatsTest(unittest.TestCase):
    def test_computes_post_transform_stats_from_parquet(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            info = {
                "robot_type": "teleavatar",
                "features": {
                    OBS_STATE: {
                        "dtype": "float32",
                        "shape": [72],
                        "names": list(TELEAVATAR_RAW_FEATURE_NAMES),
                    },
                    ACTION: {
                        "dtype": "float32",
                        "shape": [72],
                        "names": list(TELEAVATAR_RAW_FEATURE_NAMES),
                    },
                    "observation.images.head_camera": make_image_feature(1920, 3840),
                    "observation.images.left_color": make_image_feature(800, 2560),
                    "observation.images.right_color": make_image_feature(800, 2560),
                },
            }
            (root / "meta/info.json").write_text(json.dumps(info))

            state = np.stack(
                [np.arange(72, dtype=np.float32), np.arange(72, dtype=np.float32) + 2]
            )
            action = np.zeros((2, 72), dtype=np.float32)
            action[:, :7] = [np.arange(7), np.arange(7) + 2]
            action[:, 8:15] = [np.arange(7) + 10, np.arange(7) + 12]
            action[:, 39] = [2.0, 0.0]
            action[:, 47] = [-1.6, 2.0]
            table = pa.table(
                {
                    OBS_STATE: state.tolist(),
                    ACTION: action.tolist(),
                    "episode_index": [0, 0],
                }
            )
            pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")

            payload = compute_stats(root, batch_size=1)

            self.assertEqual(payload["camera_eye"], "left")
            self.assertEqual(payload["stats"][OBS_STATE]["count"], [2])
            self.assertEqual(len(payload["stats"][OBS_STATE]["mean"]), 16)
            np.testing.assert_allclose(payload["stats"][OBS_STATE]["mean"][-2:], [8, 16])
            np.testing.assert_allclose(payload["stats"][OBS_STATE]["std"][-2:], [1, 1])
            self.assertEqual(len(payload["stats"][ACTION]["mean"]), 16)
            self.assertAlmostEqual(payload["stats"][ACTION]["mean"][7], 0.05, places=6)
            self.assertAlmostEqual(payload["stats"][ACTION]["mean"][15], 0.5, places=6)
            self.assertIn("0", payload["episodes"])


if __name__ == "__main__":
    unittest.main()
