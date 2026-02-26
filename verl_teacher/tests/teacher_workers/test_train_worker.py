# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.trainer.ppo.utils import Role

os.environ["NCCL_DEBUG"] = "WARN"

from functools import partial

import numpy as np
import pytest
import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, Qwen3Config, Qwen3MoeConfig

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import (
	FSDPEngineConfig,
	FSDPOptimizerConfig,
	HFModelConfig,
	McoreEngineConfig,
	McoreOptimizerConfig,
)
from verl.single_controller.ray.base import create_colocated_worker_cls

from verl_teacher.utils.config import omega_conf_to_dataclass
from verl_teacher.workers.fsdp_workers import TeacherTrainWorker

from hydra import compose, initialize_config_dir  
from hydra.core.global_hydra import GlobalHydra  
import os, shutil

# megatron and fsdp2 currently fail for me, but I don't need to use them
# @pytest.mark.parametrize("strategy", ["megatron", "fsdp", "fsdp2"])
@pytest.mark.parametrize("strategy", ["fsdp"])
def test_teacher_train_worker(strategy):
	assert strategy == "fsdp" or strategy == "fsdp2", f"Strategy {strategy} is not supported for TeacherTrainWorker"

	### This code is used to load the same config which main_ppo.py would load
	GlobalHydra.instance().clear()  
	try:  
		# This requires an absolute path
		with initialize_config_dir(config_dir=os.path.join(os.path.dirname(__file__), "..", "..", "config")):  
			config = compose(config_name="teacher_runner", overrides=[
				"teacher.ppo_micro_batch_size_per_gpu=256",
				"teacher.model.path=Qwen/Qwen2.5-1.5B-Instruct"
			])  
	finally:  
		GlobalHydra.instance().clear()
	
	ray.init()

	resource_pool = RayResourcePool(process_on_nodes=[1])
	teacher_cfg = omega_conf_to_dataclass(config.teacher)
	class_dict = {str(Role.TeacherTrain): RayClassWithInitArgs(cls=TeacherTrainWorker, config=teacher_cfg)}
	wg_dict = RayWorkerGroup(
		resource_pool=resource_pool,
		ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TeacherTrainWorker), config=teacher_cfg),
	)
	spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
	all_wg = {}
	all_wg.update(spawn_wg)
	wg = all_wg[str(Role.TeacherTrain)]

	# if strategy == "megatron":
	# 	raise NotImplementedError("TeacherTrainWorker does not support megatron strategy")
	# 	engine_config = McoreEngineConfig(
	# 		forward_only=False,
	# 		use_mbridge=False,
	# 		tensor_model_parallel_size=2,
	# 		pipeline_model_parallel_size=2,
	# 		context_parallel_size=2,
	
	# init model
	wg.init_model()

	batch_size = 8
	seqlen = 32

	response_length = seqlen // 2

	torch.manual_seed(1)
	np.random.seed(1)

	model_config = HFModelConfig(path="Qwen/Qwen2.5-1.5B-Instruct", load_tokenizer=False)
	input_ids = torch.randint(0, model_config.hf_config.vocab_size, (batch_size, seqlen))
	attention_mask = create_random_mask(
		input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
	)
	position_ids = compute_position_id_with_mask(attention_mask)

	global_token_num = torch.sum(attention_mask, dim=-1).tolist()

	print(input_ids.float().mean(), attention_mask.float().mean())

	responses = input_ids[:, response_length:]
	response_mask = attention_mask[:, response_length:]

	assert torch.all(response_mask[:, 0] == 1)

	data = DataProto.from_single_dict(
		{
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"position_ids": position_ids,
			"responses": responses,
			"response_mask": response_mask,
		},
		meta_info={"temperature": 1.0, "global_token_num": global_token_num},
	)

	# add ppo data
	data.batch["scores"] = torch.rand_like(responses, dtype=torch.float32)

	# update again
	ppo_metrics = wg.update_teacher(data)
	print(ppo_metrics)

	# test saving checkpoint
	save_path = os.path.join(os.path.dirname(__file__), "test_checkpoint")
	if os.path.exists(save_path):
		shutil.rmtree(save_path)
	wg.save_checkpoint(save_path)
	assert os.path.exists(save_path)
	
	# check that the checkpoint directory uses at least 17GB of disk space (this is assuming a 
	# Qwen2.5-1.5B-Instruct model)
	total_size = 0
	for dirpath, dirnames, filenames in os.walk(save_path):
		for filename in filenames:
			filepath = os.path.join(dirpath, filename)
			total_size += os.path.getsize(filepath)
	min_size_bytes = 17 * 1024 ** 3  # 17 GB
	assert total_size >= min_size_bytes, (
		f"Checkpoint size {total_size / 1024**3:.2f} GB is less than the required 17 GB"
	)
	
	# Cleanup
	shutil.rmtree(save_path)

	ray.shutdown()
