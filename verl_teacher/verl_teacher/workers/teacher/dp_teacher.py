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

class DataParallelTeacher(BaseTeacher):
    def __init__(self, config, teacher_module: nn.Module, teacher_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.teacher_module = teacher_module
        self.teacher_optimizer = teacher_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Teacher use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

        self.loss_fn = nn.MSELoss(reduction="mean") if self.config.get("use_mse_loss", False) else nn.BCEWithLogitsLoss(reduction="mean")

    def _forward_micro_batch(self, micro_batch):
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
                if self.config.model.get("use_mean_pooling", False):
                    raise NotImplementedError("Mean pooling value head with remove padding has not been tested yet")
                response_length = 1
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
                
                # The output shape will be (bsz, num_outputs) where num_outputs matches the number of student models
                if not self.config.model.get("use_mean_pooling", False):
                    # The VeRL dataloader left-pads by default, so handle that here
                    scores = scores[:, -1, :]  # take the last token's output as the score
                    ### Old code before we switched to left padding:
                    # response_lengths = attention_mask.sum(dim=1, keepdim=True)
                    # # In case we have any all-padding sequences, clamp the last token index to be at least 0 to avoid negative indexing
                    # last_token_idx = (response_lengths - 1).clamp_min(0).long()
                    # # TODO: Make this work for num_outputs > 1
                    # scores = torch.gather(scores, 1, last_token_idx.unsqueeze(-1))
                    # scores = scores.squeeze(-1) # (bsz, num_outputs)
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
                scores = torch.sigmoid(scores)  # restrict to [0, 1] range
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
        # Right now pad_mask is only used in teacher_runner.py, whereas the unit tests do not include it
        if "pad_mask" in data.batch:
            select_keys.append("pad_mask")
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
                    scores= scores.to(preds.dtype)  # ensure scores and preds have the same dtype for loss computation
                    # Ignore any padding elements in the loss computation
                    if "pad_mask" in micro_batch.batch.keys():
                        if not micro_batch.batch["pad_mask"].any():
                            continue # skip this micro_batch if it is entirely padding
                        preds = preds[micro_batch.batch["pad_mask"].squeeze()]
                        scores = scores[micro_batch.batch["pad_mask"].squeeze()]
                    loss = self.loss_fn(preds, scores)
                    assert loss.numel() == 1, "Loss should be a single scalar value after reduction"

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss_scale_factor = response_mask.shape[0] / self.config.mini_batch_size
                        loss = loss * loss_scale_factor
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                        loss = loss * loss_scale_factor

                    loss.backward()

                    loss_name = "mse_loss" if self.config.get("use_mse_loss", False) else "bce_loss"
                    micro_batch_metrics.update(
                        {
                            f"teacher/{loss_name}": loss.detach().item(),
                        }
                    )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"teacher/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.teacher_optimizer.zero_grad()
        return metrics
