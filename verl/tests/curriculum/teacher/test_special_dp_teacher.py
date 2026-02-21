#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoConfig

from verl import DataProto
from verl.workers.config import FSDPTeacherConfig, FSDPOptimizerConfig
from verl.workers.config.teacher import FSDPTeacherModelCfg
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.fsdp_workers import TeacherWorker


class TestTeacherWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if not torch.distributed.is_initialized():
            # If environment variables for distributed setup are not set, we can initialize a default process group for testing
            # Only overwrite those which are not already set
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
            )

        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(cls.rank)
            cls.device = torch.device(f"cuda:{cls.rank}")
        else:
            cls.device = torch.device("cpu")

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Set up test fixtures"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp()

        config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        config.save_pretrained(self.temp_dir)

        self.config = FSDPTeacherConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            forward_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            cliprange_value=0.5,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            ulysses_sequence_parallel_size=1,
            rollout_n=1,
            optim=FSDPOptimizerConfig(lr=1e-6),
            model=FSDPTeacherModelCfg(
                path="Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_path="Qwen/Qwen2.5-0.5B-Instruct",
                fsdp_config=FSDPEngineConfig(fsdp_size=-1),
                use_remove_padding=False,
            ),
        )
        assert self.world_size <= 4 // 2

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data_for_compute_scores(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for compute_scores method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            batch_size=[batch_size],
        )

        data = DataProto(
            batch=batch, meta_info={"micro_batch_size": 2, "max_token_len": seq_len, "use_dynamic_bsz": False}
        )

        return data

    def _create_test_data_for_update_teacher(self, batch_size=2, seq_len=10, response_len=5):
        """Create test data for update_teacher method"""
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        responses = torch.randint(0, 1000, (batch_size, response_len), dtype=torch.long)
        response_mask = torch.ones(batch_size, response_len, dtype=torch.float)
        scores = torch.randn(batch_size, response_len, dtype=torch.float)
        returns = torch.randn(batch_size, response_len, dtype=torch.float)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "scores": scores,
                "returns": returns,
            },
            batch_size=[batch_size],
        )

        data = DataProto(
            batch=batch,
            meta_info={"global_token_num": [response_len] * batch_size, "batch_seqlens": [response_len] * batch_size},
        )

        return data

    def test_init_model(self):
        """Test TeacherWorker.init_model() method"""
        worker = TeacherWorker(self.config)
        worker.init_model()

        self.assertIsNotNone(worker.teacher_module)
        self.assertIsNotNone(worker.teacher_optimizer)
        self.assertIsNotNone(worker.teacher)
        self.assertIsNotNone(worker.checkpoint_manager)

    def test_compute_scores(self):
        """Test TeacherWorker.compute_scores() method"""
        worker = TeacherWorker(self.config)
        worker.init_model()

        data = self._create_test_data_for_compute_scores()

        result = worker.compute_scores(data)

        self.assertIsInstance(result, DataProto)
        self.assertIn("scores", result.batch)
        scores = result.batch["scores"]

        batch_size, response_len = 2, 5
        self.assertEqual(scores.shape, (batch_size, response_len))

        self.assertTrue(torch.isfinite(scores).all())

    def test_update_teacher(self):
        """Test TeacherWorker.update_teacher() method"""
        worker = TeacherWorker(self.config)
        worker.init_model()

        data = self._create_test_data_for_update_teacher()

        result = worker.update_teacher(data)

        self.assertIsInstance(result, DataProto)
        self.assertIn("metrics", result.meta_info)
        metrics = result.meta_info["metrics"]

        expected_keys = ["teacher/vf_loss", "teacher/vf_clipfrac", "teacher/vpred_mean", "teacher/grad_norm"]
        for key in expected_keys:
            self.assertIn(key, metrics)

        for key, value in metrics.items():
            if isinstance(value, list | tuple):
                for v in value:
                    self.assertTrue(torch.isfinite(torch.tensor(v)).all())
            else:
                self.assertTrue(torch.isfinite(torch.tensor(value)).all())

    @patch("transformers.AutoConfig.from_pretrained")
    def test_teacher_attn_implementation_override_functionality(self, mock_config_from_pretrained):
        """Test that TeacherWorker correctly uses attn_implementation from override_config"""

        # Mock the AutoConfig return value
        mock_config = Mock()
        mock_config.tie_word_embeddings = False
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_config.num_labels = 1
        mock_config_from_pretrained.return_value = mock_config

        # Test different attn_implementation values
        test_cases = [
            ("eager", "eager"),
            ("sdpa", "sdpa"),
            ("flash_attention_2", "flash_attention_2"),
            (None, "flash_attention_2"),  # Default case
        ]

        for override_value, expected_value in test_cases:
            mock_config_from_pretrained.reset_mock()

            # Create config with override_config
            config_dict = {
                "model": {
                    "path": "/test/model/path",
                    "tokenizer_path": "/test/tokenizer/path",
                    "fsdp_config": {
                        "fsdp_size": 1,
                        "param_offload": False,
                        "optimizer_offload": False,
                    },
                },
                "optim": {"lr": 1e-4, "type": "AdamW"},
                "strategy": "fsdp",
                "ppo_mini_batch_size": 1,
                "ppo_epochs": 1,
                "rollout_n": 1,
                "checkpoint": {"save_contents": [], "load_contents": []},
            }

            # Add override_config with attn_implementation if specified
            if override_value is not None:
                config_dict["model"]["override_config"] = {"attn_implementation": override_value}

            # Convert to OmegaConf
            test_config = OmegaConf.create(config_dict)

            # Test the extraction logic that should happen in TeacherWorker._build_teacher_model_optimizer
            override_config = OmegaConf.to_container(OmegaConf.create(test_config.model.get("override_config", {})))
            extracted_attn_implementation = override_config.get("attn_implementation", "flash_attention_2")

            # Verify the extraction works correctly
            self.assertEqual(
                extracted_attn_implementation,
                expected_value,
                f"Expected {expected_value}, got {extracted_attn_implementation} for override_value {override_value}",
            )

    def test_teacher_model_config_structure(self):
        """Test that teacher model config properly incorporates override settings"""

        # Test configuration scenarios
        test_scenarios = [
            {"name": "default_flash_attention", "override_config": {}, "expected_attn": "flash_attention_2"},
            {"name": "eager_override", "override_config": {"attn_implementation": "eager"}, "expected_attn": "eager"},
            {"name": "sdpa_override", "override_config": {"attn_implementation": "sdpa"}, "expected_attn": "sdpa"},
            {
                "name": "mixed_config",
                "override_config": {"attn_implementation": "eager", "dropout": 0.1, "num_labels": 1},
                "expected_attn": "eager",
            },
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Simulate the config processing logic from TeacherWorker
                override_config = scenario["override_config"]

                # Test the extraction logic
                extracted_attn = override_config.get("attn_implementation", "flash_attention_2")

                # Verify correct extraction
                self.assertEqual(extracted_attn, scenario["expected_attn"], f"Failed for scenario {scenario['name']}")

                # Verify other configs are preserved
                if "dropout" in override_config:
                    self.assertEqual(override_config["dropout"], 0.1)

    def test_teacher_backward_compatibility(self):
        """Test that TeacherWorker maintains backward compatibility with existing configurations"""

        # Test cases for backward compatibility
        compatibility_tests = [
            {"name": "no_override_config", "config": {}, "expected": "flash_attention_2"},
            {"name": "empty_override_config", "config": {"override_config": {}}, "expected": "flash_attention_2"},
            {
                "name": "other_overrides_only",
                "config": {"override_config": {"dropout": 0.1, "hidden_size": 768}},
                "expected": "flash_attention_2",
            },
        ]

        for test in compatibility_tests:
            with self.subTest(test=test["name"]):
                override_config = test["config"].get("override_config", {})
                attn_implementation = override_config.get("attn_implementation", "flash_attention_2")

                self.assertEqual(
                    attn_implementation, test["expected"], f"Backward compatibility failed for {test['name']}"
                )


if __name__ == "__main__":
    unittest.main()
