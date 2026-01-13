import asyncio
import logging
import math
from typing import Callable, Optional

import ray
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm
from transfer_queue import (
    AsyncTransferQueueClient,
    BatchMeta,
    SimpleStorageUnit,
    TransferQueueController,
    get_placement_group,
    process_zmq_server_info,
)

from verl.protocol import DataProto
from verl_tool.agent_loop.v1_hrl_agent_loop import HRLAgentLoopManager

logger = logging.getLogger(__name__)


def _run_sync(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class HRLRayTrainer:
    """HRL trainer that stages rollouts through a single TransferQueue pair."""

    def __init__(
        self,
        config,
        *,
        manager_builder: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.manager = manager_builder(config) if manager_builder else HRLAgentLoopManager(config)
        self.data_system_client = self._initialize_data_system(
            global_batch_size=config.data.train_batch_size,
            num_n_samples=config.actor_rollout_ref.rollout.n,
            role="train",
        )

    def _initialize_data_system(self, global_batch_size: int, num_n_samples: int, role: str) -> AsyncTransferQueueClient:
        cfg = OmegaConf.create(self.config, flags={"allow_objects": True})
        total_storage_size = global_batch_size * cfg.trainer.num_global_batch * num_n_samples

        storage_pg = get_placement_group(1, num_cpus_per_actor=1)
        storage_units = {
            0: SimpleStorageUnit.options(
                placement_group=storage_pg,
                placement_group_bundle_index=0,
            ).remote(storage_unit_size=math.ceil(total_storage_size))
        }
        controller_pg = get_placement_group(1, num_cpus_per_actor=1)
        controller = TransferQueueController.options(
            placement_group=controller_pg, placement_group_bundle_index=0
        ).remote(
            num_storage_units=1,
            global_batch_size=global_batch_size,
            num_global_batch=cfg.trainer.num_global_batch,
            num_n_samples=num_n_samples,
        )

        controller_info = process_zmq_server_info(controller)
        storage_infos = process_zmq_server_info(storage_units)

        ray.get([unit.register_controller_info.remote(controller_info) for unit in storage_units.values()])

        client = AsyncTransferQueueClient(
            client_id=f"HRLTrainer-{role}",
            controller_info=controller_info,
            storage_unit_infos=storage_infos,
        )
        client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=cfg)
        return client

    def _meta_to_dataproto(self, batch_meta: BatchMeta) -> DataProto:
        tensor_data = _run_sync(self.data_system_client.async_get_data(batch_meta))
        return DataProto.from_tensordict(tensor_data, meta_info=batch_meta.extra_info.copy())

    def _update_meta_with_output(self, output: DataProto, batch_meta: BatchMeta) -> BatchMeta:
        for k, v in output.meta_info.items():
            batch_meta.set_extra_info(k, v)
        if len(output) > 0:
            tensordict = output.to_tensordict()
            for key in output.meta_info.keys():
                tensordict.pop(key)
            _run_sync(self.data_system_client.async_put(data=tensordict, metadata=batch_meta))
            batch_meta.add_fields(tensordict)
        return batch_meta

    def generate_sequences(self, prompts: TensorDict, partition_id: str = "train_0") -> BatchMeta:
        _run_sync(self.data_system_client.async_put(data=prompts, partition_id=partition_id))

        batch_meta = _run_sync(
            self.data_system_client.async_get_meta(
                data_fields=list(prompts.keys()),
                batch_size=prompts.batch_size[0],
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )

        prompts_dp = self._meta_to_dataproto(batch_meta)
        hrl_output = self.manager.generate_sequences(prompts_dp)
        batch_meta = self._update_meta_with_output(hrl_output, batch_meta)

        return batch_meta

    def fit_once(self, dataloader, partition_prefix: str = "train") -> list[BatchMeta]:
        metas: list[BatchMeta] = []
        for step, prompts in enumerate(tqdm(dataloader, desc="HRL rollouts")):
            partition_id = f"{partition_prefix}_{step}"
            metas.append(self.generate_sequences(prompts, partition_id=partition_id))
            _run_sync(self.data_system_client.async_clear_partition(partition_id=partition_id))
        return metas

    def close(self) -> None:
        self.data_system_client.close()
