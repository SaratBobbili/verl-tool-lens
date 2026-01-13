import time
import uuid
from copy import deepcopy

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.utils import Role
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking
from verl.utils.rollout_skip import RolloutSkip

from verl_tool.trainer.hrl.v1_ray_trainer import HRLRayTrainer
from verl_tool.trainer.ppo.ray_trainer import AgentRayPPOTrainer
from verl.trainer.ppo.ray_trainer import (
    compute_advantage,
    compute_response_mask,
    apply_kl_penalty,
    compute_reward,
    compute_reward_async,
)


class HRLAgentRayPPOTrainer(AgentRayPPOTrainer):
    """PPO trainer that swaps rollout generation for HRL TransferQueue rollouts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hrl_trainer = HRLRayTrainer(config=self.config)
        self._hrl_partition_prefix = "train"

    def _hrl_generate_sequences(self, gen_batch: DataProto, partition_id: str) -> DataProto:
        """Route rollouts through HRL trainer while keeping meta info intact."""
        gen_batch.meta_info.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen_batch.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_batch.meta_info.setdefault("do_sample", self.config.actor_rollout_ref.rollout.do_sample)
        gen_batch.meta_info.setdefault("recompute_log_prob", False)
        gen_batch.meta_info.setdefault("validate", False)
        gen_batch.meta_info.setdefault("global_steps", self.global_steps)
        return self.hrl_trainer.generate_dataproto(gen_batch, partition_id=partition_id)

    def fit(self):
        """
        Training loop that mirrors AgentRayPPOTrainer.fit but uses HRL rollouts.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        try:
            for epoch in range(self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    metrics = {}
                    timing_raw = {}
                    partition_id = f"{self._hrl_partition_prefix}_{self.global_steps}"

                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                    is_last_step = self.global_steps >= self.total_training_steps
                    with marked_timer("step", timing_raw):
                        try:
                            with marked_timer("gen", timing_raw, color="red"):
                                gen_start = time.time()
                                gen_batch_output = self._hrl_generate_sequences(
                                    gen_batch_output, partition_id=partition_id
                                )
                                gen_duration = time.time() - gen_start
                                gen_batch_output.meta_info.setdefault("timing", {})
                                gen_batch_output.meta_info["timing"].setdefault("gen", gen_duration)
                                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))

                            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                                if self.reward_fn is None:
                                    raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                                with marked_timer("gen_max", timing_raw, color="purple"):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    baseline_partition = f"{partition_id}_baseline"
                                    try:
                                        gen_baseline_batch = self._hrl_generate_sequences(
                                            gen_baseline_batch, partition_id=baseline_partition
                                        )
                                        batch = batch.union(gen_baseline_batch)
                                        rm_scores = None
                                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                                            rm_scores = self.rm_wg.compute_rm_score(batch)
                                            batch = batch.union(rm_scores)
                                        reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                        keys_to_pop = set(gen_baseline_batch.batch.keys())
                                        if rm_scores is not None:
                                            keys_to_pop.update(rm_scores.batch.keys())
                                        batch.pop(batch_keys=list(keys_to_pop))

                                        batch.batch["reward_baselines"] = reward_baseline_tensor

                                        del rm_scores, gen_baseline_batch
                                    finally:
                                        self.hrl_trainer.clear_partition(baseline_partition)

                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)

                            if "response_mask" not in batch.batch.keys():
                                batch.batch["response_mask"] = compute_response_mask(batch)

                            if self.config.trainer.balance_batch:
                                self._balance_batch(batch, metrics=metrics)

                            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                            reward_extra_infos_dict: dict[str, list] = {}
                            with marked_timer("reward", timing_raw, color="yellow"):
                                if self.use_rm and "rm_scores" not in batch.batch.keys():
                                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                                    batch = batch.union(reward_tensor)

                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(
                                        data=batch, config=self.config, tokenizer=self.tokenizer
                                    )
                                else:
                                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)

                                if "rollout_log_probs" in batch.batch.keys():
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                            if self.use_reference_policy:
                                with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                            if self.use_critic:
                                with marked_timer("values", timing_raw, color="cyan"):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)

                            with marked_timer("adv", timing_raw, color="brown"):
                                if self.config.reward_model.launch_reward_fn_async:
                                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                batch.batch["token_level_scores"] = reward_tensor

                                if reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                                if self.config.algorithm.use_kl_in_reward:
                                    batch, kl_metrics = apply_kl_penalty(
                                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                    )
                                    metrics.update(kl_metrics)
                                else:
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                                batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                                metrics.update(is_metrics)

                                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    config=self.config.algorithm,
                                )

                            if self.use_critic:
                                with marked_timer("update_critic", timing_raw, color="pink"):
                                    critic_output = self.critic_wg.update_critic(batch)
                                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                metrics.update(critic_output_metrics)

                            if self.config.trainer.critic_warmup <= self.global_steps:
                                with marked_timer("update_actor", timing_raw, color="red"):
                                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                            if rollout_data_dir:
                                self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                        finally:
                            self.hrl_trainer.clear_partition(partition_id)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

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

                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                        self.train_dataloader.sampler.update(batch=batch)

                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        progress_bar.close()
                        return

                    if hasattr(self.train_dataset, "on_batch_end"):
                        self.train_dataset.on_batch_end(batch=batch)
        finally:
            self.hrl_trainer.close()
