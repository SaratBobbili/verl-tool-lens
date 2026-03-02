# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Teacher training and scoring with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

import time
from verl_teacher.utils.comms import Aggregator, OfflineAggregator
from verl_teacher.utils.data import zero_pad_dataproto
from verl_teacher.utils.data import extract_question_from_chat_template

def need_teacher_train(role_worker_mapping: dict[Role, WorkerType]) -> bool:
    """Given a role worker mapping, do we need teacher trainer."""
    return Role.TeacherTrain in role_worker_mapping

def need_teacher_score(role_worker_mapping: dict[Role, WorkerType]) -> bool:
    """Given a role worker mapping, do we need teacher scorer."""
    return Role.TeacherScore in role_worker_mapping

def poll_aggregator(agg, offline=False):
    if offline:
        poll_result = agg.poll_batch()
    else:
        poll_ref = agg.poll_batch.remote()
        ready, _ = ray.wait([poll_ref], timeout=0.0)
        if not ready:
            return None
        poll_result = ray.get(ready[0])
    if poll_result is None:
        return None
    bid, batch = poll_result
    return (bid, batch)

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

class TeacherRunner:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_teacher_train = need_teacher_train(self.role_worker_mapping)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        if self.config.data_source == "online":
            self.using_online_data = True
            self._create_dataloader(train_dataset, collate_fn, train_sampler)
        elif self.config.data_source == "offline":
            self.using_online_data = False
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")

    def _create_dataloader(self, train_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the dataloader for the training data to be passed to the students.
        The validation data is only loaded on the student side for evaluation, so no 
        validation dataloader is created here
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from .main_teacher import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        self.dataset = train_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.dataloader = StatefulDataLoader(
            dataset=self.dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.dataloader) >= 1, "Train dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.dataloader)}"
        )

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create teacher train worker group if needed
        if self.use_teacher_train:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherTrain)
            teacher_cfg = omega_conf_to_dataclass(self.config.teacher)
            teacher_train_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.TeacherTrain], config=teacher_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.TeacherTrain)] = teacher_train_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_teacher_train:
            self.teacher_train_wg = all_wg[str(Role.TeacherTrain)]
            self.teacher_train_wg.init_model()

        # # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        # self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        # self.actor_rollout_wg.init_model()

        # # create async rollout manager and request scheduler
        # self.async_rollout_mode = False
        # if self.config.actor_rollout_ref.rollout.mode == "async":
        #     from verl.experimental.agent_loop import AgentLoopManager

        #     self.async_rollout_mode = True
        #     self.async_rollout_manager = AgentLoopManager(
        #         config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
        #     )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `teacher_final_ckpt`
        final_save_folder = os.path.join(
            self.config.trainer.default_local_dir, f"teacher_final_ckpt"
        )

        print(f"final_save_folder: {final_save_folder}")

        teacher_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "teacher_final_ckpt")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_teacher_ckpt_to_keep=1 instead"
            )
        max_teacher_ckpt_to_keep = (
            self.config.trainer.get("max_teacher_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.teacher_train_wg.save_checkpoint(
            final_save_folder, teacher_remote_path, self.global_steps, max_ckpt_to_keep=max_teacher_ckpt_to_keep
        )

        # if self.use_critic:
        #     critic_local_path = os.path.join(final_save_folder, str(Role.Critic))
        #     critic_remote_path = (
        #         None
        #         if self.config.trainer.default_hdfs_dir is None
        #         else os.path.join(
        #             self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
        #         )
        #     )
        #     self.critic_wg.save_checkpoint(
        #         critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
        #     )

        # # save dataloader
        # local_mkdir_safe(final_save_folder)
        # dataloader_local_path = os.path.join(final_save_folder, "data.pt")
        # dataloader_state_dict = self.dataloader.state_dict()
        # torch.save(dataloader_state_dict, dataloader_local_path)

        # # latest checkpointed iteration tracker (for atomic usage)
        # local_latest_checkpointed_iteration = os.path.join(
        #     self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        # )
        # with open(local_latest_checkpointed_iteration, "w") as f:
        #     f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        # critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # # load critic
        # if self.use_critic:
        #     self.critic_wg.load_checkpoint(
        #         critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        #     )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            # if self.use_critic:
            #     self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            # if self.use_critic:
            #     self.critic_wg.stop_profile()

    def run(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        # self._load_checkpoint()

        # TODO: If loading from checkpoint, also load existing labeled data and teacher training data

        # we start from step 1
        self.global_steps += 1
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        if self.using_online_data:
            # This will automatically use the same ray namespace that was applied to ray.init() in main_teacher.py
            self.aggregator = Aggregator.options(
                name="teacher_agg",
                # lifetime="detached",      # survives if teacher driver dies (we shouldn't need this)
                get_if_exists=True,       # idempotent start
            ).remote(batch_size=self.config.data.train_batch_size)
        else:
            assert self.config.offline_data_path is not None, "offline_data_path must be specified for training on offline data"
            # self.aggregator = OfflineAggregator.remote(self.config.offline_data_path, batch_size=self.config.data.train_batch_size)
            self.aggregator = OfflineAggregator(self.config.offline_data_path, batch_size=self.config.data.train_batch_size)
        
        # TODO: This loop needs to be interrupted by some kind of signal from the students when it is time to
        # end training
        while True:
            
            ### Check if a complete training batch is available from the aggregator
            # TODO: This doesn't seem to be working for the online version; the result is None even if a batch is available
            poll_result = poll_aggregator(self.aggregator, offline=(self.config.data_source == "offline"))
            
            if poll_result is not None:
                print(f"Received batch from aggregator with ID {poll_result[0]}")  # batch ID
                # Not doing anything with batch ID right now
                batch_tuples = poll_result[1]

                if self.using_online_data:
                    # TODO: Get the actual prompts based on their IDs, and construct the DataProto
                    # We will also need to extract the question from the chat template, unless we use a separate dataset
                    # where the prompts are simplified
                    pass
                else:
                    # For offline data, the batch_tuples contain strings which must be tokenized to obtain
                    # input_ids and then batched together
                    prompts = [extract_question_from_chat_template(batch_tuple["input"]) for batch_tuple in batch_tuples]
                    # Also convert from [-1,1] scale to [0,1] scale
                    avg_scores = [(batch_tuple["avg_score"] + 1.0) / 2.0 for batch_tuple in batch_tuples]
                    tokenized = self.tokenizer(prompts, padding=True, padding_side='right', return_tensors="pt")
                    input_ids = tokenized.input_ids
                    attention_mask = tokenized.attention_mask
                    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
                    response_mask = attention_mask.clone()
                    scores = torch.tensor(avg_scores).unsqueeze(1)  # shape (batch_size, 1)

                data = DataProto.from_single_dict(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "response_mask": response_mask,
                        "scores": scores,
                    },
                    meta_info={"global_token_num": torch.sum(attention_mask, dim=-1).tolist()},
                )
                # TODO: May be better to avoid the if/else and just always call zero_pad_dataproto, 
                # which should be modified to add non_pad_indices even if no padding is needed
                if len(data) < self.config.data.train_batch_size:
                    print(
                        f"Received batch of size {len(data)}, which is smaller than the configured train batch size {self.config.data.train_batch_size}. "
                        "This batch will be zero-padded to the train batch size for teacher training."
                    )
                    data = zero_pad_dataproto(data, self.config.data.train_batch_size)
                else:
                    data.meta_info["non_pad_indices"] = list(range(len(data)))

                metrics = self.teacher_train_wg.update_teacher(data)

                # Print the MSE loss averaged over the micro-batches for this step
                mse_per_micro_batch = metrics.meta_info.get('metrics', {}).get('teacher/mse_loss', None)
                if mse_per_micro_batch:
                    mse_per_micro_batch = [mse[0] for mse in mse_per_micro_batch]  # convert list of lists to list of floats
                    avg_mse = sum(mse_per_micro_batch) / len(mse_per_micro_batch)
                    print(f"Step {self.global_steps}: Average MSE loss = {avg_mse}")
            else:
                if self.using_online_data:
                    # TODO: Get the actual prompts based on their IDs, and construct the DataProto
                    pass
                else:
                    print("All training data used")
                    break
                    # raise NotImplementedError("Handling end of training for offline data not implemented yet")

            self.global_steps += 1
            break # Temporary break

        # Save a teacher checkpoint at the end of training
        self._save_checkpoint()
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise NotImplementedError("Handling end of training not implemented yet")

        

"""for batch_dict in self.dataloader:
    metrics = {}
    timing_raw = {}

    with marked_timer("start_profile", timing_raw):
        self._start_profiling(
            not prev_step_profile and curr_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else curr_step_profile
        )
    batch: DataProto = DataProto.from_single_dict(batch_dict)

    # add uid to batch
    batch.non_tensor_batch["uid"] = np.array(
        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
    )

    # TODO: At this point we can check if there is a need for more batches on either student, and
    # if so spend time labeling more data (if we are past the warmup stage) or just directly send
    # the next batch from the dataloader
    # Until teacher scoring works, this part will be omitted, and instead the students will rely
    # on their own dataloaders

    # Now check if there is enough data from the students to do a model update

    gen_batch = batch

    # pass global_steps to trace
    gen_batch.meta_info["global_steps"] = self.global_steps

    # with marked_timer("step", timing_raw):
    #     # generate a batch
    #     with marked_timer("gen", timing_raw, color="red"):
    #         if not self.async_rollout_mode:
    #             gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
    #         else:
    #             gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

    #         timing_raw.update(gen_batch_output.meta_info["timing"])
    #         gen_batch_output.meta_info.pop("timing", None)

    #     # repeat to align with repeated responses in rollout
    #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    #     batch = batch.union(gen_batch_output)

    #     if "response_mask" not in batch.batch.keys():
    #         batch.batch["response_mask"] = compute_response_mask(batch)

    #     # compute global_valid tokens
    #     batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    #     # compute values
    #     if self.use_critic:
    #         with marked_timer("values", timing_raw, color="cyan"):
    #             values = self.critic_wg.compute_values(batch)
    #             batch = batch.union(values)

    #     # update critic
    #     if self.use_critic:
    #         with marked_timer("update_critic", timing_raw, color="pink"):
    #             critic_output = self.critic_wg.update_critic(batch)
    #         critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    #         metrics.update(critic_output_metrics)

    #     # implement critic warmup
    #     if self.config.trainer.critic_warmup <= self.global_steps:
    #         # update actor
    #         with marked_timer("update_actor", timing_raw, color="red"):
    #             batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
    #             actor_output = self.actor_rollout_wg.update_actor(batch)
    #         actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
    #         metrics.update(actor_output_metrics)

    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
    esi_close_to_expiration = should_save_ckpt_esi(
        max_steps_duration=self.max_steps_duration,
        redundant_time=self.config.trainer.esi_redundant_time,
    )
    # Check if the conditions for saving a checkpoint are met.
    # The conditions include a mandatory condition (1) and
    # one of the following optional conditions (2/3/4):
    # 1. The save frequency is set to a positive value.
    # 2. It's the last training step.
    # 3. The current step number is a multiple of the save frequency.
    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
    if self.config.trainer.save_freq > 0 and (
        self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
    ):
        if esi_close_to_expiration:
            print("Force saving checkpoint: ESI instance expiration approaching.")
        with marked_timer("save_checkpoint", timing_raw, color="green"):
            self._save_checkpoint()

    with marked_timer("stop_profile", timing_raw):
        next_step_profile = (
            self.global_steps + 1 in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self._stop_profiling(
            curr_step_profile and not next_step_profile
            if self.config.global_profiler.profile_continuous_steps
            else curr_step_profile
        )
        prev_step_profile = curr_step_profile
        curr_step_profile = next_step_profile

    steps_duration = timing_raw["step"]
    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

    # training metrics
    metrics.update(
        {
            "training/global_step": self.global_steps,
        }
    )
    # # collect metrics
    # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
    # metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
    # # TODO: implement actual tflpo and theoretical tflpo
    # n_gpus = self.resource_pool_manager.get_n_gpus()
    # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
    # # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

    # TODO: make a canonical logger that supports various backend
    logger.log(data=metrics, step=self.global_steps)

    self.global_steps += 1

    if (
        hasattr(self.config.actor_rollout_ref.actor, "profiler")
        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
    ):
        self.actor_rollout_wg.dump_memory_snapshot(
            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
        )

    # this is experimental and may be changed/removed in the future
    # in favor of a general-purpose data buffer pool
    if hasattr(self.dataset, "on_batch_end"):
        # The dataset may be changed after each training batch
        self.dataset.on_batch_end(batch=batch)
"""