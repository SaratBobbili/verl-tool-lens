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

import warnings
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import BaseModelConfig, CheckpointConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config import FSDPEngineConfig, McoreEngineConfig, HFModelConfig, OptimizerConfig

__all__ = ["TeacherConfig", "FSDPTeacherConfig", "McoreTeacherConfig", "FSDPTeacherModelCfg"]


@dataclass
class TeacherConfig(BaseConfig):
    """Configuration for teacher model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Strategy used for teacher model training (fsdp, fsdp2, megatron).
        micro_batch_size_per_gpu (int): Local per-GPU micro batch size.
        rollout_n (int): Number of rollouts per update (mirrors actor rollout_n).
        optim (Dict[str, Any]): Optimizer configuration including lr, weight_decay, etc.
        model (Dict[str, Any]): Model configuration including path, tokenizer_path, etc.
        mini_batch_size (int): PPO mini-batch size per update.
        micro_batch_size (Optional[int]): Global micro batch size (deprecated).
        use_dynamic_bsz (bool): Whether to automatically adjust batch size at runtime.
        max_token_len_per_gpu (int): Max tokens per GPU in one PPO batch.
        forward_max_token_len_per_gpu (int): Max token length per GPU in forward pass.
        epochs (int): Number of PPO epochs per batch.
        shuffle (bool): Shuffle training data across PPO epochs.
        cliprange_value (float): PPO value function clipping range.
        loss_agg_mode (str): Loss aggregation mode.
        checkpoint (Dict[str, Any]): Checkpoint configuration.
        profiler (Dict[str, Any]): Profiler configuration.
        enable (Optional[bool]): Whether to enable the teacher.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "micro_batch_size_per_gpu",
        "mini_batch_size",
        "micro_batch_size",
        "model_config",
    }

    strategy: str = MISSING
    micro_batch_size_per_gpu: Optional[int] = None
    enable: Optional[bool] = None
    trainer_enable: Optional[bool] = None
    scorer_enable: Optional[bool] = None
    rollout_n: int = 1
    mini_batch_size: int = 1
    use_dynamic_bsz: bool = False
    max_token_len_per_gpu: int = 32768
    # deprecate this
    forward_max_token_len_per_gpu: int = 32768
    infer_micro_batch_size_per_gpu: Optional[int] = None
    infer_max_token_len_per_gpu: int = 32768
    epochs: int = 1
    data_loader_seed: int = 1
    shuffle: bool = True
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    micro_batch_size: Optional[int] = None
    engine: BaseConfig = field(default_factory=BaseConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    # deprecate model to favor model_config
    model: BaseModelConfig = field(default_factory=BaseModelConfig)
    model_config: HFModelConfig = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)

    def __post_init__(self):
        """Validate teacher configuration parameters."""
        assert self.strategy != MISSING

        if self.model_config is None:
            warnings.warn("using model in Teacher Config is deprecated, please use model_config instead", stacklevel=2)
            self.model_config = self.model

        if not self.use_dynamic_bsz:
            self._check_mutually_exclusive(self.micro_batch_size, self.micro_batch_size_per_gpu, "teacher")

            if self.micro_batch_size is not None:
                if self.mini_batch_size % self.micro_batch_size != 0:
                    raise ValueError(
                        f"[teacher] mini_batch_size ({self.mini_batch_size}) must be divisible by "
                        f"micro_batch_size ({self.micro_batch_size})"
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate teacher configuration with runtime parameters.

        Args:
            n_gpus: Total number of GPUs available
            train_batch_size: Training batch size from data config
        """
        if not self.use_dynamic_bsz:
            if train_batch_size < self.mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"teacher.mini_batch_size ({self.mini_batch_size})"
                )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        param = "micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreTeacherConfig(TeacherConfig):
    """Configuration for Megatron-based teacher model training.

    The inheritance from TeacherConfig provides all base teacher configuration plus Megatron-specific settings.

    Args:
        nccl_timeout (int): NCCL timeout in seconds for distributed operations.
        megatron (Dict[str, Any]): Megatron-specific parallelism settings.
        load_weight (bool): Whether to load initial weights.
        data_loader_seed (Optional[int]): Seed for data loader.
    """

    strategy: str = "megatron"
    nccl_timeout: int = 600
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    load_weight: bool = True
    data_loader_seed: Optional[int] = None

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate Megatron teacher configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)


@dataclass
class FSDPTeacherConfig(TeacherConfig):
    """Configuration for FSDP-based teacher model training.

    The inheritance from TeacherConfig provides all base teacher configuration plus FSDP-specific settings.

    Args:
        forward_micro_batch_size (int): Forward-only batch size during inference (global).
        forward_micro_batch_size_per_gpu (int): Forward-only batch size during inference (per GPU).
        ulysses_sequence_parallel_size (int): Sequence parallelism size for Ulysses-style model parallelism.
        grad_clip (float): Gradient clipping for teacher updates.
    """

    _mutable_fields = TeacherConfig._mutable_fields | {
        "forward_micro_batch_size",
        "forward_micro_batch_size_per_gpu",
    }

    strategy: str = "fsdp"
    forward_micro_batch_size: int = 1
    forward_micro_batch_size_per_gpu: int = 1
    ulysses_sequence_parallel_size: int = 1
    grad_clip: float = 1.0

    def __post_init__(self):
        """Validate FSDP teacher configuration parameters."""
        super().__post_init__()

        if self.strategy in {"fsdp", "fsdp2"}:
            if self.ulysses_sequence_parallel_size > 1:
                if not self.model.get("use_remove_padding", False):
                    raise ValueError(
                        "When using sequence parallelism for teacher, you must enable `use_remove_padding`."
                    )

    def validate(self, n_gpus: int, train_batch_size: int):
        """Validate FSDP teacher configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size)

        if not self.use_dynamic_bsz:
            sp_size = self.ulysses_sequence_parallel_size
            if self.micro_batch_size is not None:
                if self.micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"teacher.micro_batch_size ({self.micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )


@dataclass
class FSDPTeacherModelCfg(BaseModelConfig):
    """FSDP-enabled teacher model configuration.
    Inherits base teacher settings and adds distributed-memory and LoRA options.

    Args:
        use_shm (bool): Whether to use shared memory for loading the model.
        enable_activation_offload (bool): Offload activations to CPU to reduce GPU memory usage.
        use_remove_padding (bool): Use remove-padding optimization (saves compute).
        enable_gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency.
        fsdp_config (FSDPEngineConfig): FSDP-specific configuration block.
        lora_rank (int): Set to positive value to enable LoRA (e.g., 32).
        lora_alpha (int): LoRA scaling factor.
        target_modules (Union[str, List[str]]): LoRA target modules: "all-linear" or list of layer names.
    """

    strategy: str = "fsdp"
    use_shm: bool = False
    enable_activation_offload: bool = False
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = True
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    lora_rank: int = 0
    lora_alpha: int = 16
    target_modules: str | list[str] = "all-linear"
