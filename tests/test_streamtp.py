import unittest

import torch

from lerobot.policies.smolvla2.streamtp import (
    StreamTPConfig,
    shift_action_chunk,
    streamtp_solve,
)


class StreamTPTest(unittest.TestCase):
    def test_shift_drops_executed_prefix_and_repeats_last_action(self):
        chunk = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        shifted = shift_action_chunk(chunk, 2)
        torch.testing.assert_close(shifted.flatten(), torch.tensor([2.0, 3.0, 4.0, 4.0, 4.0]))

    def test_constant_field_converges_in_two_parallel_sweeps(self):
        observed_batch_sizes = []

        def velocity(points, times):
            observed_batch_sizes.append(points.shape[0])
            return torch.ones_like(points) * 2.0

        noise = torch.zeros(1, 3, 2)
        endpoint, diagnostics = streamtp_solve(
            velocity,
            noise,
            num_steps=4,
            config=StreamTPConfig(tolerance=1e-7, max_sweeps=4, anderson_depth=0),
        )

        torch.testing.assert_close(endpoint, torch.full_like(noise, -2.0))
        self.assertTrue(diagnostics["accepted"])
        self.assertEqual(diagnostics["sweeps"], 2)
        self.assertEqual(diagnostics["expert_evaluations"], 8)
        self.assertEqual(observed_batch_sizes, [4, 4])

    def test_nonconvergence_falls_back_to_exact_sequential_euler(self):
        def velocity(points, times):
            shape = (times.shape[0],) + (1,) * (points.ndim - 1)
            return 0.25 * points + times.reshape(shape)

        noise = torch.tensor([[[1.0, -1.0], [0.5, 2.0]]])
        endpoint, diagnostics = streamtp_solve(
            velocity,
            noise,
            num_steps=5,
            config=StreamTPConfig(tolerance=1e-12, max_sweeps=1, anderson_depth=3),
            previous_actions=torch.full_like(noise, 9.0),
        )

        reference = noise.clone()
        dt = float(torch.tensor(-1.0 / 5, dtype=torch.float32))
        current_time = torch.tensor(1.0, dtype=torch.float32)
        for _ in range(5):
            reference += dt * velocity(reference, current_time.expand(1))
            current_time += dt
        torch.testing.assert_close(endpoint, reference, rtol=0, atol=0)
        self.assertTrue(diagnostics["fallback"])
        self.assertEqual(diagnostics["critical_nfe"], 6)
        self.assertEqual(diagnostics["expert_evaluations"], 10)

    def test_model_specific_exact_fallback_is_used(self):
        expected = torch.full((1, 2, 1), 7.0)
        calls = []

        def fallback(noise):
            calls.append(noise.clone())
            return expected.clone()

        endpoint, diagnostics = streamtp_solve(
            lambda points, times: torch.ones_like(points),
            torch.zeros_like(expected),
            num_steps=2,
            config=StreamTPConfig(tolerance=1e-12, max_sweeps=1),
            fallback_fn=fallback,
        )

        torch.testing.assert_close(endpoint, expected)
        self.assertEqual(len(calls), 1)
        self.assertTrue(diagnostics["fallback"])

    def test_invalid_controls_fail_before_sampling(self):
        with self.assertRaises(ValueError):
            StreamTPConfig(tolerance=0).validate()
        with self.assertRaises(ValueError):
            StreamTPConfig(max_sweeps=0).validate()
        with self.assertRaises(ValueError):
            StreamTPConfig(anderson_depth=-1).validate()

    def test_residual_ignores_padded_action_dimensions(self):
        def velocity(points, times):
            result = torch.zeros_like(points)
            result[..., 1] = 4.0
            return result

        noise = torch.zeros(1, 2, 2)
        endpoint, diagnostics = streamtp_solve(
            velocity,
            noise,
            num_steps=2,
            config=StreamTPConfig(tolerance=1e-7, max_sweeps=1),
            residual_dim=1,
        )

        self.assertTrue(diagnostics["accepted"])
        torch.testing.assert_close(endpoint[..., 0], torch.zeros_like(endpoint[..., 0]))

    def test_anderson_reduces_depth_for_affine_field(self):
        torch.manual_seed(0)
        noise = torch.randn(1, 3, 2)

        def velocity(points, times):
            return 0.5 * points + times[:, None, None]

        _, plain = streamtp_solve(
            velocity,
            noise,
            num_steps=10,
            config=StreamTPConfig(tolerance=1e-4, max_sweeps=10, anderson_depth=0),
        )
        _, accelerated = streamtp_solve(
            velocity,
            noise,
            num_steps=10,
            config=StreamTPConfig(tolerance=1e-4, max_sweeps=10, anderson_depth=3),
        )

        self.assertTrue(plain["accepted"])
        self.assertTrue(accelerated["accepted"])
        self.assertLess(accelerated["sweeps"], plain["sweeps"])


if __name__ == "__main__":
    unittest.main()
