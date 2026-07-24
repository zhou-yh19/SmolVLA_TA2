import unittest

import numpy as np

from smolvla_deploy.contracts import (
    OBS_IMAGE,
    OBS_IMAGE_2,
    OBS_IMAGE_3,
    build_state,
    map_smolvla_images,
    select_execution_actions,
    split_trigger_action,
)


class ContractsTest(unittest.TestCase):
    def test_state_is_left_then_right(self):
        state = build_state(np.arange(7), np.arange(10, 17))
        np.testing.assert_array_equal(state, [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(state.dtype, np.float32)

    def test_all_three_cameras_use_left_eye(self):
        split = {
            "head_left_eye": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_left_eye": np.ones((4, 6, 3), dtype=np.uint8),
            "right_wrist_left_eye": np.full((4, 6, 3), 2, dtype=np.uint8),
        }
        mapped = map_smolvla_images(split)
        self.assertEqual(list(mapped), [OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3])
        self.assertEqual(int(mapped[OBS_IMAGE_2][0, 0, 0]), 1)
        self.assertEqual(int(mapped[OBS_IMAGE_3][0, 0, 0]), 2)

    def test_grippers_are_direct_triggers(self):
        action = np.zeros(16, dtype=np.float32)
        action[7] = 1.0
        action[15] = 0.55
        command = split_trigger_action(action)
        self.assertEqual(command.left_trigger, 1.0)
        self.assertAlmostEqual(command.right_trigger, 0.55, places=6)

    def test_trigger_is_clipped_not_converted(self):
        action = np.zeros(16, dtype=np.float32)
        action[7] = -0.2
        action[15] = 1.2
        command = split_trigger_action(action)
        self.assertEqual(command.left_trigger, 0.0)
        self.assertEqual(command.right_trigger, 1.0)

    def test_execution_horizon_selects_chunk_prefix(self):
        chunk = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
        selected = select_execution_actions(chunk, 16)
        self.assertEqual(selected.shape, (16, 16))
        np.testing.assert_array_equal(selected, chunk[:16])


if __name__ == "__main__":
    unittest.main()
