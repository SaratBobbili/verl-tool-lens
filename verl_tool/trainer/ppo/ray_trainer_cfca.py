import uuid
from copy import deepcopy
from pprint import pprint
from typing import Optional
from collections import defaultdict


import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from .metric_util import compute_data_metrics
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.config import AlgoConfig
from .reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

from tensordict import TensorDict
from itertools import groupby

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask, apply_kl_penalty, compute_advantage
import torch.nn.functional as F

def get_segment_ids(response_mask):
    mask_change_locs = response_mask.diff(dim=1)
    # Mark each change between assistant responses and environment context, then
    # assign ids to the segments to determine which tokens will be masked at each
    # iteration
    change = torch.cat([
            # first element always starts a segment
            torch.tensor([True]*mask_change_locs.shape[0], device=response_mask.device).unsqueeze(-1),  
            mask_change_locs != 0,
        ], 
        dim=1)
    # Every segment with an even ID corresponds to a response segment.
    segment_ids = (change.cumsum(dim=1) - 1).long()  # Shape: (bs, response_length)
    return segment_ids

# This function masks out the desired turn and returns log probs for the full response sequence
# The turn index is related to the segment_ids in the following way:
# Turn 0 includes segment_ids 0 and 1, turn 1 includes segment_ids 2 and 3, and so on
# Multiply the turn index by 2 to get the first segment_id
def get_log_probs_with_mask(gen_batch_output, segment_ids, turn_idx, actor_rollout_wg):
    # Get the segment ids to mask out
    id_1 = 2*turn_idx
    id_2 = 2*turn_idx + 1
    attn_mask = gen_batch_output.batch['attention_mask'].clone()
    attn_mask[(segment_ids == id_1) | (segment_ids == id_2)] = 0
    # Compute position ids based on the new attention mask
    new_position_ids = torch.cumsum(attn_mask, dim=1) - 1
    # Get the sequence of tokens for which we want to compute log probs. This is simply
    # the tokens which follow the masked out segments.
    # For now, we will include all tokens and extract the relevant log probs later
    new_responses = gen_batch_output.batch['responses'].clone()
    # Create a new DataProto with the modified attention mask, position ids, and response mask
    masked_tensordict = TensorDict(
            {
                "input_ids": gen_batch_output.batch["input_ids"],
                "attention_mask": attn_mask,
                "position_ids": new_position_ids,
                "responses": new_responses,
            },
            batch_size=attn_mask.shape[0],
            device=gen_batch_output.batch.device,
        )
    gen_batch_output_masked = DataProto(batch=masked_tensordict, non_tensor_batch=gen_batch_output.non_tensor_batch)
    log_probs = actor_rollout_wg.compute_log_prob(gen_batch_output_masked).batch['old_log_probs']
    return log_probs

# Multiply the turn index by 2 to get the first segment_id (which is the segment
# corresponding to the model's output for that turn)
def sum_log_probs_per_turn(log_probs, start_turn_idx, max_num_turns, segment_ids, exclude_first_n=None):
    if exclude_first_n is not None:
        # Zero out the first n log probs for each turn (per row, per segment)
        for b in range(log_probs.size(0)):
            # iterate over segments present in this row and blank their first n tokens
            for seg_id in torch.unique(segment_ids[b]):
                # Only zero for even (model output) segment ids
                if seg_id < 0 or (seg_id % 2 != 0):
                    continue
                mask = segment_ids[b] == seg_id
                if not mask.any():
                    continue
                token_positions = torch.nonzero(mask, as_tuple=False).flatten()
                if token_positions.numel() == 0:
                    continue
                cutoff = token_positions[:exclude_first_n]
                log_probs[b, cutoff] = 0.0
        
    log_prob_sums = log_probs.new_zeros(log_probs.shape[0], int(segment_ids.max().item())+1)
    log_prob_sums.scatter_add_(dim=1, index=segment_ids, src=log_probs)
    # Keep only the sums corresponding to model outputs (i.e. even segment_ids) and map them
    # to their turn index
    start_id = 2*start_turn_idx
    end_id = 2*max_num_turns
    return log_prob_sums[:, range(start_id,end_id,2)]

def first_n_from_segment(A: torch.Tensor,
                         B: torch.Tensor,
                         k,
                         N: int,
                         pad_value: float = 0.0):
    """
    Extract the first N values in each row of A whose segment-id in B equals k.

    A: [bs, T] float
    B: [bs, T] int
    k: int scalar or LongTensor [bs] (per-row segment choice)
    Returns:
      out:   [bs, N]  (padded with pad_value when not enough elems)
      valid: [bs, N]  (True where out is real)
    """
    assert A.shape == B.shape and A.ndim == 2

    bs, T = A.shape
    device = A.device

    # mask: [bs, T]
    if torch.is_tensor(k):
        k = k.to(device=device)
        if k.ndim == 0:
            mask = (B == k)
        else:
            # per-row k: [bs] -> [bs, 1]
            mask = (B == k.view(-1, 1))
    else:
        mask = (B == int(k))

    # positions 0..T-1, then set non-matching to T (sentinel "very large")
    pos = torch.arange(T, device=device).view(1, T).expand(bs, T)
    pos_masked = torch.where(mask, pos, torch.full_like(pos, T))

    # take smallest N positions per row (sorted left-to-right)
    vals, idx = pos_masked.topk(N, dim=1, largest=False, sorted=True)  # [bs, N]
    valid = vals < T                                                    # [bs, N]

    out = A.gather(1, idx)                                              # [bs, N]
    out = torch.where(valid, out, torch.full_like(out, pad_value))

    return out, valid

def get_avg_log_prob_diffs_first_tokens(raw_masked_log_probs, unmasked_log_probs, response_segment_ids, num_turns, num_tokens=20):
    # We want the log probs with the first turn masked
    log_probs_first_turn_masked = raw_masked_log_probs[0]
    log_prob_diffs = unmasked_log_probs - log_probs_first_turn_masked
    # Segment id of 2 corresponds to the model's output for the second turn
    log_prob_diffs_second_turn, valid_mask = first_n_from_segment(log_prob_diffs, response_segment_ids, 2, num_tokens)
    avg_log_prob_diffs_second_turn = (log_prob_diffs_second_turn * valid_mask).sum(dim=0) / valid_mask.sum(dim=0).clamp_min(1)
    return avg_log_prob_diffs_second_turn

# The credit distribution for turn t is a softmax over the log prob differences with each
# prior turn masked (i.e. a softmax over the t'th row of turn_log_prob_differences with
# invalid elements masked out)
# In terms of LATEX document: c_{j->i} = credit_distributions[:, j, i]
def get_credit_distributions(turn_log_prob_differences, valid_turn_log_prob_mask, T=1):
    turn_log_prob_differences[valid_turn_log_prob_mask == 0] = -torch.inf
    # TODO: Make temperature configurable
    credit_distributions = F.softmax(turn_log_prob_differences/T, dim=2)
    return credit_distributions

# This function computes the overall distribution over which the outcome advantage will be
# distributed to prior turns. For example, if there are a total of 3 turns for rollout i, then
# overall_distribution_weights[i, 1] = credit_distributions[i, 1, 1] and
# overall_distribution_weights[i, 0] = credit_distributions[i, 1, 0] + credit_distributions[i, 0, 0]*overall_distribution_weights[i, 1]
# NOTE: The indexing is counterintuitive since index 0 of dimension 1 of credit_distributions 
# corresponds to the second turn, but index 0 of dimension 2 corresponds to the first turn
# (This is because dimension 1 corresponds to the turn whose log probs are being compared,
# while dimension 2 corresponds to the turn which is being masked). Furthermore,
# Dimension 1 of overall_distribution_weights corresponds to the turn which is receiving 
# credit, hence index 0 corresponds to the first turn.
def distribute_credit(credit_distributions):
    # For simplicity, expand credit_distributions to include the first turn in dimension 1.
    # These elements will all be zero so they won't affect the computation of weights
    credit_distributions = torch.cat([credit_distributions.new_zeros(credit_distributions.shape[0], 1, credit_distributions.shape[2]), credit_distributions], dim=1)
    # Initialize the overall distribution weights (size: bs x max_num_turns)
    overall_distribution_weights = credit_distributions.new_zeros(credit_distributions.shape[0], credit_distributions.shape[1])
    overall_distribution_weights[:, -1] = 1.0  # The last turn gets full credit for the outcome
    # Start with the second-to-last turn and work backwards
    for t in range(credit_distributions.shape[1]-2, -1, -1):
        credit_products = torch.multiply(credit_distributions[:, (t+1):, t], overall_distribution_weights[:, (t+1):])
        overall_distribution_weights[:, t] = torch.sum(credit_products, dim=1)
    # Get the final distribution
    return F.softmax(overall_distribution_weights[:, :-1], dim=1)

def groups_from_index_array(index: np.ndarray):
    """Generate groups of indices based on the provided index array.

    Args:
        index (np.ndarray): An array where each element indicates the group ID for the corresponding data point.

    Returns:
        groups (List[torch.Tensor]): A list of tensors, each containing the indices of data points belonging to the same group.
        idx_groups (defaultdict(list)): A dictionary mapping group IDs to lists of indices.
    """
    idx_groups = defaultdict(list)
    for i, s in enumerate(index):
        idx_groups[s].append(i)
    groups = [torch.tensor(idxs, dtype=torch.long) for idxs in idx_groups.values()]
    return groups, idx_groups

def compute_cfca_outcome_advantage(
        token_level_rewards: torch.Tensor, 
        credit_weights: torch.Tensor, 
        response_mask: torch.Tensor, 
        index: torch.Tensor, 
        epsilon: float = 1e-6,
        norm_adv_by_std_in_grpo=True, 
        config: Optional[AlgoConfig]=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage with counterfactual credit assignment.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        credit_weights: `(torch.Tensor)`
            shape is (bs, max_num_turns-1) where max_num_turns is the maximum number of turns over the batch
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # Get outcome rewards
    scores = token_level_rewards.sum(dim=-1)
    # We need to first group the scores and credit_weights based on index.
    # Note that we will need to restore the original order when we return the advantages
    groups, idx_groups = groups_from_index_array(index)
    scores_grouped = [scores[g] for g in groups]
    if credit_weights is not None:
        credit_weights_grouped = [credit_weights[g] for g in groups]
    
    # Use the standard GRPO advantage computation to get the advantages for the final turns,
    # then distribute them to prior turns using the credit weights
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    
    # If credit weights were not calculated (e.g. not enough turns to apply CFCA) or every
    # outcome advantage is zero, revert to standard GRPO
    if credit_weights is not None and (scores != 0).any():
        # Replace nan values with 1.0 to distribute the outcome advantage equally in cases
        # where distribution weights were not calculated
        credit_weights = torch.nan_to_num(credit_weights, nan=1.0)
        # Append a tensor with all ones for the final turn (which gets full credit for the outcome)
        credit_weights = torch.cat([credit_weights, torch.ones((credit_weights.shape[0], 1), device=credit_weights.device)], dim=1)
        # Compute final per-turn advantages
        advantages_per_turn = credit_weights * scores.unsqueeze(-1)

        segment_ids = get_segment_ids(response_mask)
        # Every segment with an even ID corresponds to a response segment. We insert a zero at every odd index
        # in final_scores to align the true rewards with segment_ids
        expanded_advantages_per_turn = torch.zeros((advantages_per_turn.shape[0], int(segment_ids.max().item())+1), device=advantages_per_turn.device)
        expanded_advantages_per_turn[:, 0::2] = advantages_per_turn
        advantages = expanded_advantages_per_turn.gather(1, segment_ids)
        return advantages, advantages
    else:
        scores = scores.unsqueeze(-1) * response_mask
        return scores, scores

def compute_advantage_cfca(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    assert adv_estimator == AdvantageEstimator.GRPO, f"adv_estimator must be GRPO for CFCA, got {adv_estimator}"
    # Initialize the mask for GRPO calculation
    grpo_calculation_mask = data.batch["response_mask"]

    advantages, returns = compute_cfca_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        credit_weights=data.batch.get("credit_weights", None),
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data

class RayPPOTrainerCFCA(RayPPOTrainer):
    """RayPPOTrainer from verl with Counterfactual Credit Assignment (CFCA) support.

    This trainer extends the standard RayPPOTrainer to compare the probabilities of turns 
    with and without other turns being masked.

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
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    
                    """ ####################### START OF CFCA-SPECIFIC CODE BLOCK ####################### """
                    raw_masked_log_probs = []
                    with marked_timer("masked_fp", timing_raw, color="magenta"):
                        num_turns = gen_batch_output.non_tensor_batch['__num_turns__']
                        max_num_turns = num_turns.max().item()
                        response_mask = gen_batch_output.batch['response_mask']
                        # Get segment ids for the response tokens (different ids for each model
                        # output and environment response)
                        segment_ids = get_segment_ids(response_mask)  # Shape: (bs, response_length)

                        input_ids = gen_batch_output.batch["input_ids"] # This is only used for debugging
                        # NOTE: segment_ids gets modified in the loop below, but we want to preserve the
                        # original values in response_segment_ids (which excludes the prompt and left padding) 
                        # so we clone the tensor
                        response_segment_ids = segment_ids.clone()

                        # Initialize the tensor which will store the probabilities of each turn's actions with
                        # a prior action mask. The first dimension corresponds to the rollout. The second 
                        # dimension corresponds to the turn which the probability is being calculated for. The 
                        # third dimension corresponds to the turn being masked. Any element for which there is
                        # no probability to calculate is set to -1 to indicate that it should be ignored. This
                        # includes any element for which the third dimension is <= to the first dimension 
                        masked_turn_log_probs = -1*torch.ones((gen_batch_output.batch['input_ids'].shape[0], max_num_turns, max_num_turns-1), device=gen_batch_output.batch['input_ids'].device)
                        # Mask to indicate which elements of masked_turn_log_probs can be used in computation
                        valid_turn_log_prob_mask = torch.ones_like(masked_turn_log_probs, dtype=torch.int32)
                        # Ignore turns which are greater than the max number of turns for the given rollout
                        rollout_turns_mask = torch.arange(max_num_turns)[None, :] >= torch.from_numpy(num_turns)[:, None]
                        valid_turn_log_prob_mask[rollout_turns_mask, :] = 0
                        for turn_idx in range(max_num_turns-1):
                            if max_num_turns == 1:
                                break
                            active_rows = (num_turns > turn_idx+1)
                            # Ignore turns up to and including turn_idx since their probabilities
                            # cannot be influenced by current masked turn
                            influence_mask = torch.arange(max_num_turns) <= turn_idx
                            valid_turn_log_prob_mask[:, influence_mask, turn_idx] = 0
                            if active_rows.any():
                                # The attention mask includes the prompt (and any left padding), so we need to pad 
                                # segment_ids accordingly
                                pad_size = gen_batch_output.batch['attention_mask'].shape[1] - segment_ids.shape[1]
                                if pad_size > 0:
                                    segment_ids = torch.cat([
                                        -1*torch.ones((segment_ids.shape[0], pad_size), device=segment_ids.device),
                                        segment_ids
                                    ], dim=1)
                                # We also set segment_ids to -1 for rows that are not active, so that we don't mask anything
                                segment_ids[~active_rows] = -1
                                
                                log_probs = get_log_probs_with_mask(gen_batch_output, segment_ids, turn_idx, self.actor_rollout_wg)
                                raw_masked_log_probs.append(log_probs)

                                # Get the log prob sums for each turn after turn_ids
                                masked_turn_log_probs[active_rows, (turn_idx+1):, turn_idx] = sum_log_probs_per_turn(log_probs[active_rows],
                                                                                                               turn_idx+1,
                                                                                                               max_num_turns,
                                                                                                               response_segment_ids[active_rows],
                                                                                                               exclude_first_n=None)
                                
                            else:
                                break
                    """ ####################### END OF CFCA-SPECIFIC CODE BLOCK ####################### """

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
                        """ ####################### START OF CFCA-SPECIFIC CODE BLOCK ####################### """
                        # TODO: If the order of data in batch gets changed by _balance_batch, we need to 
                            # account for it here
                        # See how the log prob differences for the first few tokens of the second turn behave
                        # when masking the first turn
                        if len(raw_masked_log_probs) > 0:
                            avg_per_token_log_prob_diffs = get_avg_log_prob_diffs_first_tokens(
                                raw_masked_log_probs,
                                old_log_prob.batch['old_log_probs'],
                                response_segment_ids,
                                num_turns,
                                num_tokens=20
                            )
                            for token_idx in range(3):
                                logger.log(
                                    data={f"cfca/avg_LP_diff_token_{token_idx}_2nd_turn": avg_per_token_log_prob_diffs[token_idx].detach().item()},
                                    step=self.global_steps,
                                )

                        if max_num_turns > 2:
                            old_log_prob_tensor = old_log_prob.batch["old_log_probs"]
                            # Shape will be [bs, max_num_turns-1]
                            unmasked_turn_log_probs = sum_log_probs_per_turn(old_log_prob_tensor, 1, max_num_turns, response_segment_ids, exclude_first_n=None).unsqueeze(-1)
                            # We exclude the first turn from this computation since there are no turns 
                            # before it to mask and hence no valid difference to take
                            turn_log_prob_differences = unmasked_turn_log_probs - masked_turn_log_probs[:,1:,:]
                            # Need to exclude the first turn from the mask as well
                            valid_turn_log_prob_mask = valid_turn_log_prob_mask[:,1:,:]

                            for indx in range(2):
                                valid_third_turn_log_prob_differences = turn_log_prob_differences[num_turns >= 3,1,indx] * valid_turn_log_prob_mask[num_turns >= 3,1,indx]
                                logger.log(
                                    data={f"cfca/third_turn_LP_difference_with_turn_{indx}_masked": valid_third_turn_log_prob_differences.nanmean().detach().item()},
                                    step=self.global_steps,
                                )
                            
                            # For each rollout and each turn (starting at the 3rd turn), average the log prob differences
                            # when the prior turn is masked verses when the turn before it is masked
                            if max_num_turns >= 3:
                                prior_turn_masked = []
                                other_turn_masked = []
                                for i in range(turn_log_prob_differences.shape[0]):
                                    for j in range(1,turn_log_prob_differences.shape[1]):
                                        if valid_turn_log_prob_mask[i,j,j]:
                                            prior_turn_masked.append(turn_log_prob_differences[i,j,j])
                                        if valid_turn_log_prob_mask[i,j,j-1]:
                                            other_turn_masked.append(turn_log_prob_differences[i,j,j-1])
                                prior_turn_masked_mean = np.mean(prior_turn_masked)
                                other_turn_masked_mean = np.mean(other_turn_masked)

                            # Determine how to distribute each turn's credit to previous turns
                            # Use the temperature parameter T to control the sharpness of the distribution
                            credit_distributions = get_credit_distributions(turn_log_prob_differences, valid_turn_log_prob_mask, T=self.config.algorithm.get("cfca_temperature", 1.0))
                            # Determine the how the credit for the outcome will be distributed to each turn
                            overall_distribution = distribute_credit(credit_distributions)
                            batch.batch['credit_weights'] = (max_num_turns-1)*overall_distribution
                            for indx in range(batch.batch['credit_weights'].shape[1]):
                                logger.log(
                                    data={f"cfca/turn_{indx}_weight_mean": batch.batch['credit_weights'][num_turns >= 3,indx].nanmean().detach().item()},
                                    step=self.global_steps,
                                )
                            # Log the average difference in credit weights between turn 1 and turn 0 for rollouts with at least 3 turns
                            logger.log(
                                    data={f"cfca/credit_weight_diff": (batch.batch['credit_weights'][num_turns >= 3,1] - batch.batch['credit_weights'][num_turns >= 3,0]).nanmean().detach().item()},
                                    step=self.global_steps,
                                )

                        # NOTE: If there are not enough turns to apply the credit assignment method,
                        # don't add anything to the batch object; in the advantage computation function
                        # it will revert to the standard GRPO advantage if no credit_distribution key 
                        # is found in batch.batch   

                        """ ####################### END OF CFCA-SPECIFIC CODE BLOCK ####################### """
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

                        batch = compute_advantage_cfca(
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