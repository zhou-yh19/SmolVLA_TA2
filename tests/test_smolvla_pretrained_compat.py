#!/usr/bin/env python3

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import safetensors.torch
import torch
from torch import nn

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla2.configuration_smolvla2 import SmolVLA2Config, SmolVLAConfig
from lerobot.policies.smolvla2.modeling_smolvla2 import load_smolvla
from teleavatar_v2.smolvla_deploy.policy_runtime import SmolVLARuntime, load_policy_config


class _Normalizer(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.register_buffer("buffer_mean", torch.full((2,), value))


class _TinyPolicy(nn.Module):
    def __init__(self, normalization_value: float = 1.0):
        super().__init__()
        self.model = nn.Linear(2, 2, bias=False)
        self.normalize_inputs = _Normalizer(normalization_value)
        self.normalize_targets = _Normalizer(normalization_value)
        self.unnormalize_outputs = _Normalizer(normalization_value)


class _TinyTiedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, 2, bias=False)
        self.output = nn.Linear(2, 2, bias=False)
        self.output.weight = self.embedding.weight


class SmolVLAPretrainedCompatibilityTest(unittest.TestCase):
    def test_lerobot_smolvla_config_alias_accepts_known_runtime_fields(self):
        payload = {
            "type": "smolvla",
            "use_peft": False,
            "rtc_config": None,
            "compile_model": True,
            "compile_mode": "max-autotune",
            "pretrained_revision": "test-revision",
            "pretrained_path": "lerobot/smolvla_base",
        }
        with TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            config = PreTrainedConfig.from_pretrained(directory)

        self.assertIsInstance(config, SmolVLAConfig)
        self.assertEqual(config.type, "smolvla")
        self.assertTrue(config.compile_model)
        self.assertEqual(config.pretrained_revision, "test-revision")
        self.assertTrue(config.state_to_prefix)

    def test_training_config_uses_path_as_new_finetuning_initialization(self):
        payload = {
            "type": "smolvla",
            "push_to_hub": False,
            "optimizer_lr": 1e-4,
            "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        }
        with TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            config = TrainPipelineConfig(
                dataset=DatasetConfig(repo_id="test/dataset"),
                policy=SmolVLA2Config(push_to_hub=False),
                output_dir=Path(directory, "output"),
            )
            argv = [
                "train.py",
                f"--policy.path={directory}",
                "--policy.load_vlm_weights=false",
                "--policy.optimizer_lr=2.5e-5",
            ]
            with patch.object(sys, "argv", argv):
                config.validate()

        self.assertIsInstance(config.policy, SmolVLAConfig)
        self.assertEqual(config.policy.pretrained_path, directory)
        self.assertFalse(config.policy.load_vlm_weights)
        self.assertEqual(config.optimizer.lr, 2.5e-5)

    def test_training_config_rejects_path_with_resume(self):
        config = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="test/dataset"),
            policy=SmolVLA2Config(push_to_hub=False),
            resume=True,
        )
        with patch.object(sys, "argv", ["train.py", "--policy.path=checkpoint"]):
            with self.assertRaisesRegex(ValueError, "cannot be used together"):
                config.validate()

    def test_deployment_loads_official_and_repository_checkpoint_types(self):
        for policy_type, expected_class in (
            ("smolvla", SmolVLAConfig),
            ("smolvla2", SmolVLA2Config),
        ):
            with self.subTest(policy_type=policy_type), TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text(
                    json.dumps({"type": policy_type, "push_to_hub": False})
                )
                config = load_policy_config(directory)
                self.assertIsInstance(config, expected_class)

    def test_peft_checkpoint_fails_with_actionable_error(self):
        payload = {"type": "smolvla", "use_peft": True}
        with TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(NotImplementedError, "Merge the adapter"):
                PreTrainedConfig.from_pretrained(directory)

    def test_strict_loader_accepts_target_dataset_normalization(self):
        source = _TinyPolicy()
        target = _TinyPolicy()
        checkpoint = {
            "model._orig_mod.weight": torch.full_like(source.model.weight, 3.0),
        }

        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            load_smolvla(
                target,
                checkpoint_path,
                checkpoint_keys_mapping="model._orig_mod.//model.",
            )

        torch.testing.assert_close(target.model.weight, torch.full_like(target.model.weight, 3.0))
        torch.testing.assert_close(target.normalize_inputs.buffer_mean, torch.ones(2))

    def test_deployment_restores_uninitialized_checkpoint_normalization(self):
        target = _TinyPolicy(normalization_value=float("inf"))
        checkpoint = {
            "model.weight": torch.full_like(target.model.weight, 3.0),
            "normalize_inputs.buffer_mean": torch.full((2,), 2.0),
            "normalize_targets.buffer_mean": torch.full((2,), 4.0),
            "unnormalize_outputs.buffer_mean": torch.full((2,), 6.0),
        }

        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            load_smolvla(target, checkpoint_path)

        torch.testing.assert_close(target.normalize_inputs.buffer_mean, torch.full((2,), 2.0))
        torch.testing.assert_close(target.normalize_targets.buffer_mean, torch.full((2,), 4.0))
        torch.testing.assert_close(target.unnormalize_outputs.buffer_mean, torch.full((2,), 6.0))

    def test_deployment_rejects_checkpoint_without_required_normalization(self):
        target = _TinyPolicy(normalization_value=float("inf"))
        checkpoint = {"model.weight": torch.full_like(target.model.weight, 3.0)}

        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "Missing keys.*normalize_inputs"):
                load_smolvla(target, checkpoint_path)

    def test_deployment_guard_rejects_nonfinite_normalization(self):
        with self.assertRaisesRegex(RuntimeError, "infinity or NaN.*normalize_inputs"):
            SmolVLARuntime._validate_normalization_statistics(
                _TinyPolicy(normalization_value=float("inf"))
            )

    def test_strict_loader_accepts_omitted_shared_tensor_alias(self):
        target = _TinyTiedPolicy()
        checkpoint = {"embedding.weight": torch.full_like(target.embedding.weight, 4.0)}
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            load_smolvla(target, checkpoint_path)

        torch.testing.assert_close(target.embedding.weight, torch.full_like(target.embedding.weight, 4.0))
        torch.testing.assert_close(target.output.weight, torch.full_like(target.output.weight, 4.0))

    def test_strict_loader_rejects_missing_model_weight(self):
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file({"normalize_inputs.buffer_mean": torch.ones(2)}, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "Missing keys.*model.weight"):
                load_smolvla(_TinyPolicy(), checkpoint_path)

    def test_strict_loader_rejects_unexpected_weight(self):
        source = _TinyPolicy()
        checkpoint = {
            "model.weight": source.model.weight.detach().clone(),
            "unexpected.weight": torch.ones(1),
        }
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "Unexpected keys.*unexpected.weight"):
                load_smolvla(_TinyPolicy(), checkpoint_path)

    def test_strict_loader_rejects_shape_mismatch(self):
        checkpoint = {"model.weight": torch.ones(3, 2)}
        with TemporaryDirectory() as directory:
            checkpoint_path = Path(directory, "model.safetensors")
            safetensors.torch.save_file(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "Shape mismatches.*model.weight"):
                load_smolvla(_TinyPolicy(), checkpoint_path)


if __name__ == "__main__":
    unittest.main()
