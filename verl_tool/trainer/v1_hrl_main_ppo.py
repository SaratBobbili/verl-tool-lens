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
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import is_cuda_available
from verl_tool.trainer.main_ppo import TaskRunner as BaseTaskRunner
from omegaconf import open_dict, OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_hrl(config)


def run_hrl(config) -> None:
    """Initialize Ray and launch hierarchical RL task runners."""
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    hrl_config = OmegaConf.to_container(config.get("hrl", {}), resolve=True)
    num_task_runners = hrl_config.get("num_task_runners", 1)

    profiler_cfg = config.get("global_profiler", {})
    use_nsys = (
        is_cuda_available
        and profiler_cfg.get("tool") == "nsys"
        and profiler_cfg.get("steps") is not None
        and len(profiler_cfg.get("steps", [])) > 0
    )
    nsight_runtime_env = {}
    if use_nsys:
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            profiler_cfg.global_tool_config.nsys.controller_nsight_options
        )
        nsight_runtime_env = {"nsight": nsight_options}

    runner_cls = HRLTaskRunner
    if nsight_runtime_env:
        runner_cls = HRLTaskRunner.options(runtime_env=nsight_runtime_env)

    task_runners = []
    for idx in range(num_task_runners):
        task_runners.append(runner_cls.remote())

    ray.get(
        [
            runner.run.remote(config=config, runner_rank=idx, world_size=num_task_runners)
            for idx, runner in enumerate(task_runners)
        ]
    )

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # HRL coordinator runs light logic; adjust if needed
class HRLTaskRunner:
    """Ray remote class coordinating hierarchical RL."""

    def __init__(self):
        self.runner_rank = None
        self.world_size = None

    def run(self, config, runner_rank: int = 0, world_size: int = 1):
        """Entry point for each HRL task runner."""
        from pprint import pprint

        self.runner_rank = runner_rank
        self.world_size = world_size
        print(
            f"HRLTaskRunner rank={runner_rank}/{world_size} "
            f"hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        with open_dict(config):
            config.actor_rollout_ref.rollout.agent = config.actor_rollout_ref.rollout.get("agent", {})
            config.actor_rollout_ref.rollout.agent.default_agent_loop = "v1_hrl_selector_expert"
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class = (
                "verl_tool.agent_loop.v1_hrl_agent_loop.HRLAgentLoopManager"
            )

        # Delegate to the base TaskRunner (PPO-level) with HRL agent loop enabled.
        base_runner = BaseTaskRunner.options(name=f"ppo_task_runner_{runner_rank}").remote()
        return ray.get(base_runner.run.remote(config))


if __name__ == "__main__":
    main()
