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
"""
Implement a multiprocess Teacher
"""

import logging
import os

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_, get_fsdp_full_state_dict, fsdp_version
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

from verl_teacher.workers.teacher import BaseTeacher

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def compute_mse_loss(
    preds: torch.Tensor,
    scores: torch.Tensor,
):
    """
    Compute the MSE loss for regression with the teacher model.

    Loosely based on compute_value_loss in core_algos

    Args:
        preds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        scores (torch.FloatTensor):
            Ground truth scores, shape (batch_size, response_length).

    Returns:
        mse_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated MSE loss.
    """
    mse_loss = (preds - scores) ** 2
    return mse_loss.mean()

class DataParallelTeacher(BaseTeacher):
    def __init__(self, config, teacher_module: nn.Module, teacher_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.teacher_module = teacher_module
        self.teacher_optimizer = teacher_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Teacher use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

        # version bookkeeping (currently only available with the AI-generated save/load below)
        self.version = 0

    def _forward_micro_batch(self, micro_batch):
        # TODO: Setting response_length to 1 will give us the desired result of returning the last token's score,
        # but in the future we will remove response_length entirely and probably use pooling
        response_length = 1
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs
            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.teacher_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.teacher_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    scores_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    scores_rmpad = output.logits
                    scores_rmpad = scores_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    scores_rmpad = gather_outputs_and_unpad(
                        scores_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                scores = pad_input(scores_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                scores = scores[:, -response_length - 1 : -1]
            else:
                output = self.teacher_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.teacher_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    scores = output[2]
                else:
                    scores = output.logits
                scores = scores[:, -response_length - 1 : -1].squeeze(-1)
            return scores

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        assert self.teacher_optimizer is not None, "Teacher optimizer is None; did you init in inference-only mode?"

        if isinstance(self.teacher_module, FSDP):
            grad_norm = self.teacher_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.teacher_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.teacher_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.teacher_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.teacher_optimizer.zero_grad()
        else:
            self.teacher_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp teacher", logger=logger)
    def compute_scores(self, data: DataProto) -> torch.Tensor:
        self.teacher_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = (
            ["input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["input_ids", "attention_mask", "position_ids"]
        )
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        scores_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                scores = self._forward_micro_batch(model_inputs)
            scores_lst.append(scores)
        scores = torch.concat(scores_lst, dim=0)
        
        if use_dynamic_bsz:
            scores = restore_dynamic_batch(scores, batch_idx_list)
        return scores

    @GPUMemoryLogger(role="dp teacher", logger=logger)
    def update_teacher(self, data: DataProto):
        assert self.teacher_optimizer is not None, "Teacher optimizer is None; this DP teacher is inference-only."
        # make sure we are in training mode
        self.teacher_module.train()
        metrics = {}

        # TODO: Remove response_mask
        select_keys = ["input_ids", "response_mask", "attention_mask", "position_ids", "scores"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.mini_batch_size)

        for _ in range(self.config.epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.mini_batch_size // self.config.micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)

                self.teacher_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    scores = model_inputs["scores"]

                    preds = self._forward_micro_batch(model_inputs)
                    
                    loss = compute_mse_loss(preds, scores)

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.mini_batch_size
                        loss = loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = loss * loss_scale_factor

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "teacher/mse_loss": loss.detach().item(),
                        }
                    )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"teacher/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.teacher_optimizer.zero_grad()
        return metrics
