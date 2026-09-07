import unittest

import numpy as np

from smolvla_deploy.contracts import (
    ACTION_DIM,
    OBS_IMAGE,
    OBS_IMAGE_2,
    OBS_IMAGE_3,
    STATE_DIM,
    STATE_LAYOUT,
    build_state,
    gripper_position,
    map_smolvla_images,
    pad_to_width,
    resolve_checkpoint_width,
    select_execution_actions,
    split_trigger_action,
    validate_state_layout,
)


class ContractsTest(unittest.TestCase):
    def test_state_is_left_then_right(self):
        state = build_state(np.arange(7), np.arange(10, 17), 0.25, 0.75)
        np.testing.assert_array_equal(state, [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 0.25, 0.75])
        self.assertEqual(state.dtype, np.float32)

    def test_gripper_feedback_requires_a_finite_single_position(self):
        self.assertEqual(gripper_position([0.5]), 0.5)
        for invalid in ([], [0, 1], [float("nan")], [float("inf")]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gripper_position(invalid)
        with self.assertRaises(ValueError):
            build_state(np.zeros(7), np.zeros(7), float("nan"), 0.0)

    def test_layout_marker_rejects_legacy_and_unknown_checkpoints(self):
        validate_state_layout(STATE_LAYOUT)
        for layout in (None, "", "arms14", "future_layout"):
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, "state layout"):
                validate_state_layout(layout)

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


class CheckpointWidthTest(unittest.TestCase):
    """A single-dataset run stores 16/16; a multi-dataset run pads both to 32."""

    def test_unpadded_widths_are_accepted(self):
        self.assertEqual(resolve_checkpoint_width("observation.state", [STATE_DIM], STATE_DIM, 32), STATE_DIM)
        self.assertEqual(resolve_checkpoint_width("action", [ACTION_DIM], ACTION_DIM, 32), ACTION_DIM)

    def test_padded_widths_are_accepted(self):
        self.assertEqual(resolve_checkpoint_width("observation.state", [32], STATE_DIM, 32), 32)
        self.assertEqual(resolve_checkpoint_width("action", [32], ACTION_DIM, 32), 32)

    def test_width_between_real_and_max_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_width("action", [20], ACTION_DIM, 32)

    def test_width_above_max_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_checkpoint_width("action", [64], ACTION_DIM, 32)

    def test_padding_is_zeros_and_keeps_the_real_prefix(self):
        state = build_state(np.arange(7), np.arange(10, 17), 0.25, 0.75)
        padded = pad_to_width(state, 32)
        self.assertEqual(padded.shape, (32,))
        self.assertEqual(padded.dtype, np.float32)
        np.testing.assert_array_equal(padded[:STATE_DIM], state)
        np.testing.assert_array_equal(padded[STATE_DIM:], np.zeros(32 - STATE_DIM))

    def test_padding_to_the_same_width_is_a_no_op(self):
        state = build_state(np.zeros(7), np.zeros(7), 0.0, 0.0)
        np.testing.assert_array_equal(pad_to_width(state, STATE_DIM), state)

    def test_padding_never_truncates(self):
        with self.assertRaises(ValueError):
            pad_to_width(np.zeros(32, dtype=np.float32), STATE_DIM)


if __name__ == "__main__":
    unittest.main()
