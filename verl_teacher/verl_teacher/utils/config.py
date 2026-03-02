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

from dataclasses import is_dataclass
from typing import Any, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See VeRL's trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
    """
    # number of GPUs total
    n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

    # # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    # def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
    #     """Validate mutually exclusive micro batch size configuration options.

    #     Ensures that users don't set both deprecated micro_batch_size and
    #     the new micro_batch_size_per_gpu parameters simultaneously.

    #     Args:
    #         mbs: Deprecated micro batch size parameter value.
    #         mbs_per_gpu: New micro batch size per GPU parameter value.
    #         name (str): Configuration section name for error messages.

    #     Raises:
    #         ValueError: If both parameters are set or neither is set.
    #     """
    #     settings = {
    #         "reward_model": "micro_batch_size",
    #         "actor_rollout_ref.ref": "log_prob_micro_batch_size",
    #         "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
    #     }

    #     if name in settings:
    #         param = settings[name]
    #         param_per_gpu = f"{param}_per_gpu"

    #         if mbs is None and mbs_per_gpu is None:
    #             raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

    #         if mbs is not None and mbs_per_gpu is not None:
    #             raise ValueError(
    #                 f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
    #                 f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
    #             )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    teacher_config = omega_conf_to_dataclass(config.teacher)
    teacher_config.validate(n_gpus, config.data.train_batch_size)

    print("[validate_config] All configuration checks passed successfully!")
