import asyncio
import logging
import os
import time
import uuid
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from tqdm.asyncio import tqdm

from verl.protocol import DataProto
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl_tool.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopWorker,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    RewardManagerWorker,
    _DummyConfig,
    _InternalAgentLoopOutput,
    _agent_loop_registry,
    get_trajectory_info,
    register,
)
from verl_tool.agent_loop.agent_loop import rollout_trace_attr
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils import hf_tokenizer, hf_processor
from verl_tool.agent_loop.v1_hrl_token_bridge import TokenBridge
from verl_tool.agent_loop.v1_hrl_data_controller import HRLDataSharingController
from verl_tool.agent_loop.v1_hrl_replay import HRLRoleReplay, HRL_ROLE_REPLAY_NAME

logger = logging.getLogger(__name__)


@register("v1_hrl_selector_expert")
class HierarchicalAgentLoop(AgentLoopBase):
    """Selector/expert alternating loop that always finishes on selector."""

    def __init__(
        self,
        *args,
        selector_server_manager: AsyncLLMServerManager,
        selector_tokenizer,
        expert_server_managers: list[AsyncLLMServerManager],
        data_controller=None,
        expert_agent_loop_name: str = "verltool_agent",
        expert_trainer_config: _DummyConfig | None = None,
        expert_max_turns: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.selector_server_manager = selector_server_manager
        self.selector_tokenizer = selector_tokenizer
        self.expert_server_managers = expert_server_managers
        self.data_controller = data_controller
        self.expert_agent_loop_name = expert_agent_loop_name
        self.expert_trainer_config = expert_trainer_config
        self.expert_max_turns = expert_max_turns

        hrl_cfg: DictConfig = OmegaConf.create({}) if self.config.get("hrl") is None else self.config.hrl
        self.max_turns = hrl_cfg.get("max_turns", 6)
        self.max_total_tokens = hrl_cfg.get(
            "max_total_tokens", self.config.actor_rollout_ref.rollout.response_length
        )
        self.stop_tokens = hrl_cfg.get("stop_tokens", [self.selector_tokenizer.eos_token_id])
        self.selector_sampling = hrl_cfg.get("selector_sampling", {})
        self.num_experts = int(hrl_cfg.num_experts)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Alternate selector and expert generations until selector stops or limits hit."""
        prompt_ids = list(kwargs.get("raw_prompt_ids", []))
        source_text = kwargs.get("full_prompts") or kwargs.get("prompt") or ""
        if not source_text and prompt_ids:
            source_text = self.selector_tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if not source_text:
            source_text = ""

        bridge = TokenBridge(prompt_text=source_text)
        role_spans: list[dict[str, Any]] = []
        num_turns = 0
        stop_reason = "max_turns"
        request_id = str(uuid.uuid4())
        if self.data_controller is not None:
            await self.data_controller.start.remote(request_id, bridge.prompt_text)

        selector_params = {
            "temperature": self.selector_sampling.get("temperature", sampling_params.get("temperature", 1.0)),
            "top_p": self.selector_sampling.get("top_p", sampling_params.get("top_p", 1.0)),
            "repetition_penalty": self.selector_sampling.get(
                "repetition_penalty", sampling_params.get("repetition_penalty", 1.0)
            ),
            "logprobs": self.selector_sampling.get("logprobs", sampling_params.get("logprobs", False)),
        }

        t0 = time.time()
        while num_turns < self.max_turns:
            selector_prompt_ids = bridge.encode(self.selector_tokenizer)
            selector_output: TokenOutput = await self.selector_server_manager.generate(
                request_id=request_id,
                prompt_ids=selector_prompt_ids,
                sampling_params=selector_params,
                image_data=kwargs.get("multi_modal_data", {}).get("image") if kwargs.get("multi_modal_data") else None,
                audio_data=kwargs.get("multi_modal_data", {}).get("audio") if kwargs.get("multi_modal_data") else None,
            )
            selector_tokens = selector_output.token_ids
            remaining_sel = bridge.remaining_budget(self.selector_tokenizer, self.max_total_tokens)
            take = min(len(selector_tokens), remaining_sel)
            selector_tokens = selector_tokens[:take]
            selector_start = bridge.response_token_length(self.tokenizer)
            delta_text = bridge.append_from_tokens(self.selector_tokenizer, selector_tokens)
            selector_end = bridge.response_token_length(self.tokenizer)
            if selector_end > selector_start:
                role_spans.append({"role": "selector", "start": selector_start, "end": selector_end})
            if self.data_controller is not None:
                await self.data_controller.push_selector.remote(
                    request_id, delta_text, selector_output.routed_experts
                )
            num_turns += 1

            if take == 0 or self._should_stop(selector_tokens):
                stop_reason = selector_output.stop_reason or "selector_stop"
                break
            if bridge.remaining_budget(self.selector_tokenizer, self.max_total_tokens) <= 0:
                stop_reason = "max_tokens"
                break

            routed_expert = self._choose_expert(selector_output)
            if routed_expert is None:
                continue

            expert_manager = self.expert_server_managers[routed_expert]
            expert_prompt_ids = bridge.encode(self.tokenizer)
            expert_output = await self._run_expert_agent_turn(
                expert_manager=expert_manager,
                prompt_ids=expert_prompt_ids,
                sampling_params=sampling_params,
                multi_modal_data=kwargs.get("multi_modal_data"),
                validate=kwargs.get("validate", False),
            )
            expert_tokens = expert_output.response_ids
            remaining_exp = bridge.remaining_budget(self.tokenizer, self.max_total_tokens)
            take = min(len(expert_tokens), remaining_exp)
            expert_tokens = expert_tokens[:take]
            expert_start = bridge.response_token_length(self.tokenizer)
            delta_text = bridge.append_from_tokens(self.tokenizer, expert_tokens)
            expert_end = bridge.response_token_length(self.tokenizer)
            if expert_end > expert_start:
                role_spans.append(
                    {"role": f"expert_{routed_expert}", "start": expert_start, "end": expert_end}
                )
            if self.data_controller is not None:
                await self.data_controller.push_expert.remote(request_id, routed_expert, delta_text)
            num_turns += 1

            if bridge.remaining_budget(self.tokenizer, self.max_total_tokens) <= 0:
                stop_reason = "max_tokens"
                break

        elapsed = time.time() - t0
        if self.data_controller is not None:
            await self.data_controller.finalize.remote(request_id)

        prompt_ids, response_ids, response_mask = bridge.finalize_tokens(
            self.tokenizer, self.max_total_tokens
        )
        metrics = AgentLoopMetrics(generate_sequences=elapsed, tool_calls=0.0)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            multi_modal_data=kwargs.get("multi_modal_data"),
            reward_score=None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "stop_reason": stop_reason,
                "transcript_text": bridge.transcript_text,
                "role_spans": role_spans,
            },
        )

    def _should_stop(self, tokens: list[int]) -> bool:
        if not tokens:
            return False
        return any(tok in self.stop_tokens for tok in tokens)

    def _choose_expert(self, selector_output: TokenOutput) -> Optional[int]:
        routed = selector_output.routed_experts
        if routed is None:
            return None
        if isinstance(routed, list) and len(routed) > 0:
            try:
                return int(routed[0]) % self.num_experts
            except Exception:
                return None
        try:
            return int(routed) % self.num_experts
        except Exception:
            return None

    async def _run_expert_agent_turn(
        self,
        *,
        expert_manager: AsyncLLMServerManager,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: Optional[dict[str, Any]],
        validate: bool,
    ) -> AgentLoopOutput:
        """Run a single-turn expert agent loop using the base agent implementation."""
        assert self.expert_agent_loop_name in _agent_loop_registry, (
            f"Expert agent loop {self.expert_agent_loop_name} not registered; "
            f"available: {_agent_loop_registry.keys()}"
        )
        agent_loop_config = _agent_loop_registry[self.expert_agent_loop_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=self.expert_trainer_config,
            server_manager=expert_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        output: AgentLoopOutput = await agent_loop.run(
            sampling_params,
            raw_prompt_ids=prompt_ids,
            multi_modal_data=multi_modal_data,
            validate=validate,
        )
        return output


@ray.remote(num_cpus=1)
class HRLAgentLoopWorker:
    """Agent loop worker that carries selector and expert servers."""

    def __init__(
        self,
        config: DictConfig,
        expert_handle_groups: list[list[ray.actor.ActorHandle]],
        selector_server_handles: list[ray.actor.ActorHandle],
        rm_executor=None,
        data_controller=None,
        role_replay=None,
    ):
        self.selector_server_handles = selector_server_handles
        # Flatten all expert handles for base init (uses first group); we keep groups separately
        flat_expert_handles = [h for group in expert_handle_groups for h in group]

        # Base async agent-loop setup (mirrors AgentLoopWorker.__init__).
        self.config = config
        self.rm_executor = rm_executor
        self.server_manager = AsyncLLMServerManager(config, flat_expert_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        self.role_replay = role_replay
        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path, self.rm_executor)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

        run_time_context = ray.get_runtime_context()
        self.name = run_time_context.get_actor_name() or "unnamed"

        selector_path = copy_to_local(config.hrl.selector.model.path)
        self.selector_tokenizer = hf_tokenizer(selector_path, trust_remote_code=True)
        self.selector_processor = hf_processor(selector_path, trust_remote_code=True)
        self.selector_server_manager = AsyncLLMServerManager(config.hrl.selector, selector_server_handles)
        # Build per-expert server managers (all equal config)
        self.expert_server_managers = [
            AsyncLLMServerManager(config.actor_rollout_ref.rollout, handles) for handles in expert_handle_groups
        ]
        self.data_controller = data_controller
        self.expert_agent_loop_name = config.hrl.get("expert_agent_loop_name", "verltool_agent")
        self.expert_max_turns = int(config.hrl.get("expert_max_turns", 1))

        expert_cfg = OmegaConf.create(OmegaConf.to_container(self.config, resolve=False))
        with open_dict(expert_cfg):
            expert_cfg.actor_rollout_ref.rollout.agent = expert_cfg.actor_rollout_ref.rollout.get("agent", {})
            expert_cfg.actor_rollout_ref.rollout.agent.default_agent_loop = self.expert_agent_loop_name
            expert_cfg.actor_rollout_ref.agent = expert_cfg.actor_rollout_ref.get("agent", {})
            expert_cfg.actor_rollout_ref.agent["max_turns"] = self.expert_max_turns
            expert_cfg.actor_rollout_ref.agent["val_max_turns"] = self.expert_max_turns
        self.expert_trainer_config = _DummyConfig(config=expert_cfg)

        # Optional per-worker concurrency control (mirror base worker semantics).
        self.max_concurrent_trajectories = self.config.actor_rollout_ref.agent.get("max_concurrent_trajectories", None)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from HRL agent loop."""
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        # Optional semaphore to bound per-worker concurrency.
        if self.max_concurrent_trajectories is not None:
            print(f"HRL Agent Worker {self.name} using semaphore with max concurrency {self.max_concurrent_trajectories}")
            semaphore = asyncio.Semaphore(self.max_concurrent_trajectories)

            def semaphore_wrapper(func):
                async def wrapper(*args, **kwargs):
                    async with semaphore:
                        return await func(*args, **kwargs)

                return wrapper

        else:

            def semaphore_wrapper(func):
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                return wrapper

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(semaphore_wrapper(self._run_agent_loop)(sampling_params, trajectory_info[i], **kwargs)))

        print(f"HRL Agent Worker {self.name} launching {len(tasks)} tasks...")
        outputs = await tqdm.gather(*tasks, desc=f"HRL Agent Worker {self.name} Looping", total=len(tasks))
        print(f"HRL Agent Worker {self.name} finished {len(tasks)} tasks.")
        output = self._postprocess(outputs)
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> list[_InternalAgentLoopOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                selector_server_manager=self.selector_server_manager,
                selector_tokenizer=self.selector_tokenizer,
                expert_server_managers=self.expert_server_managers,
                data_controller=self.data_controller,
                expert_agent_loop_name=self.expert_agent_loop_name,
                expert_trainer_config=self.expert_trainer_config,
                expert_max_turns=self.expert_max_turns,
            )
            kwargs["validate"] = trajectory["validate"]
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_logprobs = None
            if output.response_logprobs is not None:
                pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
                response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            position_ids = compute_position_id_with_mask(attention_mask)

            enable_async_reward = (
                self.rm_executor is not None and self.config.reward_model.enable_resource_pool
            ) or not self.config.reward_model.enable
            if output.reward_score is None and enable_async_reward:
                batch = TensorDict(
                    {
                        "prompts": prompt_output["input_ids"],
                        "responses": response_output["input_ids"],
                        "attention_mask": attention_mask,
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                }
                extra_fields = {}
                for key, val in output.extra_fields.items():
                    extra_fields[key] = np.array([val], dtype=object)

                non_tensor_batch.update(extra_fields)
                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                    meta_info={"global_steps": trajectory["step"], "validate": trajectory["validate"]},
                )
                result = await self.reward_manager_worker.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            base_extra_fields = dict(output.extra_fields)
            spans = base_extra_fields.get("role_spans") or [{"role": "selector", "start": 0, "end": len(output.response_mask)}]
            unpadded_response_length = int(response_output["attention_mask"].sum().item())
            role_outputs: list[_InternalAgentLoopOutput] = []
            for span in spans:
                start = max(0, int(span.get("start", 0)))
                end = min(unpadded_response_length, int(span.get("end", start)))
                if end <= start:
                    continue

                role_mask = torch.zeros_like(response_mask_output["input_ids"])
                role_mask[..., start:end] = response_mask_output["input_ids"][..., start:end]
                role_mask = role_mask * response_output["attention_mask"]

                role_extra_fields = dict(base_extra_fields)
                role_extra_fields["model_role"] = span.get("role", "selector")

                role_outputs.append(
                    _InternalAgentLoopOutput(
                        prompt_ids=prompt_output["input_ids"],
                        response_ids=response_output["input_ids"],
                        input_ids=input_ids,
                        position_ids=position_ids,
                        response_mask=role_mask,
                        attention_mask=attention_mask,
                        response_logprobs=response_logprobs,
                        multi_modal_inputs=None,
                        multi_modal_data=output.multi_modal_data,
                        reward_score=output.reward_score,
                        num_turns=output.num_turns,
                        metrics=output.metrics,
                        extra_fields=role_extra_fields,
                    )
                )

            if not role_outputs:
                fallback_fields = dict(base_extra_fields)
                fallback_fields["model_role"] = "selector"
                role_outputs.append(
                    _InternalAgentLoopOutput(
                        prompt_ids=prompt_output["input_ids"],
                        response_ids=response_output["input_ids"],
                        input_ids=input_ids,
                        position_ids=position_ids,
                        response_mask=response_mask,
                        attention_mask=attention_mask,
                        response_logprobs=response_logprobs,
                        multi_modal_inputs=None,
                        multi_modal_data=output.multi_modal_data,
                        reward_score=output.reward_score,
                        num_turns=output.num_turns,
                        metrics=output.metrics,
                        extra_fields=fallback_fields,
                    )
                )

            return role_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput | list[_InternalAgentLoopOutput]]) -> DataProto:
        flat_inputs: list[_InternalAgentLoopOutput] = []
        for item in inputs:
            if isinstance(item, list):
                flat_inputs.extend(item)
            else:
                flat_inputs.append(item)

        data = AgentLoopWorker._postprocess(self, flat_inputs)

        if self.role_replay is not None and len(data) > 0:
            try:
                self.role_replay.push_batch.remote(data)
            except Exception as exc:
                logger.warning(f"Failed to push HRL batch to replay: {exc}")
        return data


class HRLAgentLoopManager(AgentLoopManager):
    """Agent loop manager that spins up selector and expert rollout servers."""

    def __init__(self, config: DictConfig, worker_group=None, rm_wg=None):
        self.selector_rollout_replicas = []
        self.selector_server_handles = []
        self.selector_server_addresses = []
        self.data_controller = HRLDataSharingController.remote()
        hrl_cfg = config.get("hrl", {})
        role_replay_name = hrl_cfg.get("role_replay_name") if hasattr(hrl_cfg, "get") else None
        self.role_replay_name = role_replay_name or HRL_ROLE_REPLAY_NAME
        self.role_replay = HRLRoleReplay.options(name=self.role_replay_name).remote()
        super().__init__(config=config, worker_group=worker_group, rm_wg=rm_wg)
        self.expert_handle_groups = []

    def _initialize_llm_servers(self):
        super()._initialize_llm_servers()

        selector_cfg = self.config.hrl.selector
        rollout_world_size = (
            selector_cfg.rollout.tensor_model_parallel_size
            * selector_cfg.rollout.data_parallel_size
            * selector_cfg.rollout.pipeline_model_parallel_size
        )
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        num_replicas = max(1, world_size // max(1, rollout_world_size))

        rollout_replica_class = get_rollout_replica_class(selector_cfg.rollout.name)
        rollout_config = selector_cfg.rollout
        model_config = selector_cfg.model
        self.selector_rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        self._run_all([replica.init_standalone() for replica in self.selector_rollout_replicas])
        self.selector_server_handles = [replica._server_handle for replica in self.selector_rollout_replicas]
        self.selector_server_addresses = [replica._server_address for replica in self.selector_rollout_replicas]

        # partition expert server handles into groups (experts)
        num_experts = int(self.config.hrl.num_experts)
        if num_experts > len(self.server_handles):
            raise ValueError(
                f"num_experts {num_experts} exceeds available expert servers {len(self.server_handles)}"
            )
        group_sizes = [len(self.server_handles) // num_experts] * num_experts
        for i in range(len(self.server_handles) % num_experts):
            group_sizes[i] += 1
        idx = 0
        self.expert_handle_groups = []
        for g in group_sizes:
            self.expert_handle_groups.append(self.server_handles[idx : idx + g])
            idx += g

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        expected_num_workers = self.config.trainer.nnodes
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        num_workers = max(expected_num_workers, num_workers)
        logger.warning(f"Initializing {num_workers} HRL agent loop workers...")

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                HRLAgentLoopWorker.options(
                    name=f"hrl_agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.expert_handle_groups,
                    self.selector_server_handles,
                    self.rm_executor,
                    self.data_controller,
                    self.role_replay,
                )
            )

