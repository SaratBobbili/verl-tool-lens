
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from tensordict import TensorDict

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class RayPPOTrainerIGPO(RayPPOTrainer):
    """RayPPOTrainer from verl with Importance-Guided Policy Optimization (IGPO) support.

    This trainer extends the standard RayPPOTrainer to compute information gain across turns.

    See RayPPOTrainer definition for details on initialization args
    """

    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
    
    def fit(self):
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
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
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

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
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

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                
                with marked_timer("step", timing_raw):
                    ############################### Begin IGPO-specific code ###############################
                    # The ground-truth answers for batch i are stored in batch_dict['reward_model'][i]['ground_truth']['target']
                    # This returns a list, but so far I have ony seen questions with a single version of the ground-truth answer
                    batch_size = len(batch_dict['reward_model'])
                    ground_truths = [batch_dict['reward_model'][i]['ground_truth']['target'] for i in range(batch_size)]
                    assert all(len(gt)==1 for gt in ground_truths), "Got multiple valid ground-truth answers for some samples"
                    ground_truths = [gt[0] for gt in ground_truths]  # Extract the single ground-truth answer per sample

                    # Note that the special tokens used here by IGPO are the same for verl-tool, and they are also using a Qwen model. 
                    # Thus, I use a similar string for the pseudo-response; however, I changed the formatting a bit to match
                    # What the Qwen-2.5-1.5B-Instruct model outputs normally
                    pseudo_resps_with_gt = [self.tokenizer(f"<think> Now there's enough information to answer.</think>\n<answer>\n{ground_truth}\n</answer><|im_end|>", return_tensors="pt")['input_ids'] for ground_truth in ground_truths]
                    len_st = len(self.tokenizer("<think> Now there's enough information to answer.</think>\n<answer>\n", return_tensors="pt")['input_ids'].tolist()[0])
                    len_ed = len(self.tokenizer("\n</answer><|im_end|>", return_tensors="pt")['input_ids'].tolist()[0])
                    gt_end_indices = [] # Stores the index corresponding to the last ground-truth token for each sample in the batch

                    # Pad pseudo_resps_with_gt to max_len and stack; also expand to match group size
                    max_len = max([resp_with_gt.size(1) for resp_with_gt in pseudo_resps_with_gt])
                    pad_id = self.tokenizer.pad_token_id
                    padded_resps = [
                        torch.nn.functional.pad(resp_with_gt, (0, max_len - resp_with_gt.size(1)), value=pad_id).repeat(self.config.actor_rollout_ref.rollout.n, 1)
                        for resp_with_gt in pseudo_resps_with_gt
                    ]
                    pseudo_resps_with_gt_stacked = torch.cat(padded_resps, dim=0) # Shape: (B, max_len)
                    
                    for i, resp_with_gt in enumerate(pseudo_resps_with_gt):
                        # The end idx will account for padding
                        idx = [len_st, resp_with_gt.shape[1] - len_ed - 1]
                        assert idx[1]-idx[0]+1 == len(self.tokenizer(ground_truths[i], return_tensors="pt")['input_ids'].tolist()[0]) # TODO: This is just a sanity check
                        gt_end_indices.append(idx[1])
                    gt_end_indices = torch.tensor(gt_end_indices).repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0)  # Shape: (B,)
                    ############################### End IGPO-specific code ###############################

                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    ############################### Begin IGPO-specific code ###############################
                    # Split the generated outputs before each turn to compute the correct answer probabilities
                    # Just need to take gen_batch_output.batch['input_ids'] and split each row at the correct
                        # token IDs
                    with marked_timer("gt_probs", timing_raw, color="blue"):
                        # The prompt begins with token 151644 and ends with token 198. 
                        # The response always begins with 151644, 77091, 198 (where 198 is newline)
                        turn_start_seq = torch.tensor([151644, 77091, 198]).to(gen_batch_output.batch['input_ids'].device)
                        ### Truncate so that we can add the ground-truth answer at the end
                        # First remove any columns which only contain padding tokens (since the tensors are padded to max length)
                        x = gen_batch_output.batch['input_ids']
                        non_pad_mask = (x != pad_id).any(dim=0)  # (T,) - True for columns with at least one non-pad token
                        x = x[:, non_pad_mask]  # Filter out pad-only columns
                        B, T = x.shape
                        L = turn_start_seq.shape[0]
                        # Create sliding windows: (B, T-L+1, L)
                        windows = x.unfold(1, L, 1)
                        # Compare each window to turn_start_seq
                        matches = (windows == turn_start_seq).all(dim=2)   # (B, T-L+1)
                        # Find first occurrence index in each row
                        found = matches.any(dim=1)
                        # Index of the first match start (undefined if not found, so we fix that below)
                        first_match_start = matches.float().argmax(dim=1)   # (B,)
                        # Length we want to keep for each row:
                        #   if found: start_index + L  (truncate right AFTER the pattern)
                        #   if not found: keep full row (length = T)
                        lengths = torch.where(
                            found,
                            first_match_start + L,
                            torch.full_like(first_match_start, T)
                        )   # shape: (B,)
                        # We need a single max length for the output tensor
                        max_len = int(lengths.max().item())
                        # Build mask of "valid positions" for each row
                        # mask[b, t] = True if t < lengths[b]
                        idx = torch.arange(max_len, device=x.device)       # (max_len,)
                        mask = idx.unsqueeze(0) < lengths.unsqueeze(1)     # (B, max_len)
                        # Initialize output with padding
                        out = torch.full((B, max_len), pad_id,
                                        dtype=x.dtype, device=x.device)
                        out[mask] = x[:, :max_len][mask]

                        ### Add the ground-truth answers at the end of each sequence
                        pseudo_resps_with_gt_stacked = pseudo_resps_with_gt_stacked.to(out.device)
                        # TODO: when different prefixes are batched together, make sure there is no right-padding left
                            # This doesn't seem to be an issue on the first turn
                        pseudo_rollouts = torch.cat([out, pseudo_resps_with_gt_stacked], dim=1)
                        ### Get the logprobs of the ground-truth answers
                        # Naively form the attention mask and position_ids (position_ids for each sequence will not be affected by left-padding)
                        attention_mask = (pseudo_rollouts != pad_id).long()
                        position_ids = torch.cumsum(attention_mask, dim=1) - 1
                        # Only the length of the response mask matters here; if the length is L, then only the
                        # logprobs for the last L tokens are returned
                        responses = torch.ones((pseudo_rollouts.shape[0], pseudo_resps_with_gt_stacked.shape[1]), dtype=torch.long)
                        pseudo_rollout_tensordict = TensorDict(
                            {
                                "input_ids": pseudo_rollouts,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                                "responses": responses,
                            },
                            batch_size=pseudo_rollouts.shape[0],
                            device=pseudo_rollouts.device,
                        )
                        pseudo_rollout_DP = DataProto.from_tensordict(pseudo_rollout_tensordict, meta_info=gen_batch_output.meta_info)
                        pseudo_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_rollout_DP).batch['old_log_probs']
                        # TODO: Make this work for varying prefix and varying length of ground-truth answer
                        # TODO: Do we get logprobs for padding tokens? If not, the end index should be adjusted (if we need to use it)
                        # gt_log_probs = pseudo_log_probs[:, len_st:pseudo_log_probs.shape[1]-len_ed]
                        gt_log_probs = pseudo_log_probs.clone()
                        gt_log_probs[:, :len_st] = 0.0  # Zero out logprobs before the start of the ground-truth answer
                        mask = torch.arange(gt_log_probs.shape[1]).unsqueeze(0) > gt_end_indices.unsqueeze(1)
                        gt_log_probs = gt_log_probs.masked_fill(mask, 0.0)  # Zero out logprobs beyond the ground-truth answer
                        gt_probs = torch.exp(gt_log_probs.sum(dim=1))  # Sum logprobs to get total prob for the ground-truth answer
                    ############################### End IGPO-specific code ###############################

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
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
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
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
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
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
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)