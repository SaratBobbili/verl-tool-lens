import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.dataset.rl_dataset import collate_fn
from verl_tool.trainer.hrl import HRLRayTrainer
from verl_tool.trainer.main_ppo import create_rl_dataset, create_rl_sampler


@hydra.main(config_path="config", config_name="v1_ppo_trainer_hrl", version_base=None)
def main(config):
    run_hrl_ppo(config)


def run_hrl_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        runtime_env["worker_process_setup_hook"] = "verl_tool.trainer.main_ppo.worker_setup_hook"
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = HRLTaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class HRLTaskRunner:
    """Ray entry that runs HRL rollouts via HRLRayTrainer."""

    def run(self, config):
        from pprint import pprint

        print(f"HRLTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        validate_config(config=config, use_reference_policy=False, use_critic=False)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from torch.utils.data import DataLoader

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=train_sampler,
            num_workers=config.data.get("dataloader_num_workers", 0),
            drop_last=True,
            collate_fn=collate_fn,
        )

        trainer = HRLRayTrainer(config=config)
        trainer.fit_once(train_dataloader, partition_prefix="train")
        trainer.close()


if __name__ == "__main__":
    main()
