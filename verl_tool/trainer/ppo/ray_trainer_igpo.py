
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

def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    rewards: (batch_size, T) tensor
    gamma: scalar discount factor
    mask: (batch_size, T) tensor with 1 for valid steps and 0 for padding

    returns: (batch_size, T) tensor where returns[:, t] = sum_{k=t}^{T-1} gamma^{k-t} * rewards[:, k]
    """
    B, T = rewards.shape

    # gamma^0, gamma^1, ..., gamma^{T-1}
    discounts = gamma ** torch.arange(T, device=rewards.device)

    # Multiply rewards by gamma^t so that a simple cumsum over reversed time works
    # (batch, T) -> (batch, T)
    rew_discounted = rewards * discounts  # r_t * gamma^t

    # Reverse time dimension
    rew_discounted_rev = torch.flip(rew_discounted, dims=[1])

    # Cumulative sum over reversed time
    returns_rev = torch.cumsum(rew_discounted_rev, dim=1)

    # Flip back and divide out gamma^t to get standard returns
    returns = torch.flip(returns_rev, dims=[1]) / discounts

    return returns

def insert_at_first_false(A, B, mask):
    bs, _ = B.shape

    first_false = (~mask).float().argmax(dim=1)
    has_false = (~mask).any(dim=1)
    first_false = torch.where(has_false, first_false, torch.full_like(first_false, -1))

    rows = has_false.nonzero(as_tuple=True)[0]
    out = B.clone()
    out[rows, first_false[rows]] = A[rows, 0]
    return out

def get_group_indices(N: int, G: int, x: int):
    """ N is total batch size, G is group size, x is index in [0, N) """
    assert N % G == 0
    assert 0 <= x < N

    group_start = (x // G) * G
    return list(range(group_start, group_start + G))

# Use this to match how IGPO is described in their paper
def normalize_group(group_indx, scores_grouped, info_gain_rewards_grouped, info_gain_mask_grouped, epsilon, norm_adv_by_std_in_grpo):
    # Put all the valid rewards for a group into a single list for computing statistics
    # info_gain_mask indicates which elements are actual turn scores and which should be ignored 
    # (due to fewer actual turns). When the mask is applied, the remaining elements are also flattened
    all_group_scores = scores_grouped[group_indx].tolist() + info_gain_rewards_grouped[group_indx][info_gain_mask_grouped[group_indx].bool()].tolist()
    # Compute mean and std for scores
    mean_score = torch.mean(torch.tensor(all_group_scores))
    std_score = torch.std(torch.tensor(all_group_scores))
    # Temporarily concatenate the final scores with the info gain rewards for normalization
    rollout_scores = torch.cat([scores_grouped[group_indx].unsqueeze(-1), info_gain_rewards_grouped[group_indx]], dim=1)

    # Normalize scores
    if norm_adv_by_std_in_grpo:
        normalized_scores = (rollout_scores - mean_score) / (std_score + epsilon)
    else:
        normalized_scores = rollout_scores - mean_score
    return normalized_scores

# This version normalizes the outcome rewards and info gain rewards separately
def normalize_group_separately(group_indx, scores_grouped, info_gain_rewards_grouped, info_gain_mask_grouped, epsilon, norm_adv_by_std_in_grpo):
    # Compute mean and std for scores
    mean_score = torch.mean(scores_grouped[group_indx])
    std_score = torch.std(scores_grouped[group_indx])
    # Normalize scores
    if norm_adv_by_std_in_grpo:
        normalized_scores = (scores_grouped[group_indx].unsqueeze(-1) - mean_score) / (std_score + epsilon)
    else:
        normalized_scores = scores_grouped[group_indx].unsqueeze(-1) - mean_score

    # Compute mean and std for info gain rewards
    valid_info_gains = info_gain_rewards_grouped[group_indx][info_gain_mask_grouped[group_indx].bool()]
    if len(valid_info_gains) > 0:
        mean_info_gain = torch.mean(valid_info_gains)
        std_info_gain = torch.std(valid_info_gains)
        # Normalize info gain rewards
        if norm_adv_by_std_in_grpo:
            normalized_info_gains = (info_gain_rewards_grouped[group_indx] - mean_info_gain) / (std_info_gain + epsilon)
        else:
            normalized_info_gains = info_gain_rewards_grouped[group_indx] - mean_info_gain
    else:
        # If there are no valid info gains, just set normalized info gains to zero
        normalized_info_gains = torch.zeros_like(info_gain_rewards_grouped[group_indx])
    # Concatenate normalized scores and normalized info gains
    normalized_scores = torch.cat([normalized_scores, normalized_info_gains], dim=1)
    return normalized_scores

def compute_igpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    info_gain_rewards: torch.Tensor,
    info_gain_mask: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    gamma: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for IGPO.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        info_gain_rewards: `(torch.Tensor)`
            shape is (bs, max_turns-1)
        info_gain_mask: `(torch.Tensor)`
            shape is (bs, max_turns-1)
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

    scores = token_level_rewards.sum(dim=-1)
    # For some reason, the groups are interleaved here (as seen from index), so we cannot just split
    # the tensors immediately. We need to first group the scores and info_gain_rewards based on index.
    # It seems that the order of index can be arbitrary
    # Note that we will need to restore the original order when we return the advantages
    groups, idx_groups = groups_from_index_array(index)
    
    scores_grouped = [scores[g] for g in groups]
    info_gain_rewards_grouped = [info_gain_rewards[g] for g in groups]
    info_gain_mask_grouped = [info_gain_mask[g] for g in groups]

    group_normalized_scores = []
    group_normalized_info_gains = []
    for i in range(len(groups)):
        if config.get("normalize_info_gain_and_outcome_separately", False):
            normalized_scores = normalize_group_separately(i, scores_grouped, info_gain_rewards_grouped, info_gain_mask_grouped, epsilon, norm_adv_by_std_in_grpo)
        else:
            normalized_scores = normalize_group(i, scores_grouped, info_gain_rewards_grouped, info_gain_mask_grouped, epsilon, norm_adv_by_std_in_grpo)
        group_normalized_scores.append(normalized_scores[:, 0].unsqueeze(-1))
        # Here we also zero out the elements of info_gain_rewards where the mask is False so that they do not
        # interfere with later calculations
        group_normalized_info_gains.append(normalized_scores[:, 1:] * info_gain_mask_grouped[i].long())

    # Restore the original batch order
    final_scores = torch.empty((scores.shape[0], 1), device=scores.device)
    final_info_gain = torch.empty((scores.shape[0], info_gain_rewards.shape[1]), device=scores.device)
    for i, idxs in enumerate(idx_groups.values()):
        final_scores[idxs] = group_normalized_scores[i]
        final_info_gain[idxs] = group_normalized_info_gains[i]

    # Create a new tensor that puts the rewards in the correct order (info gains followed by outcome reward)
    # with padding to account for different number of turns. We first extend info_gain_mask with a column
    # that is all False, then insert the outcome reward at the first position where the mask is False on 
    # each row. This means that a row with n turns will have the outcome reward at index n, followed by padding.
    # Likewise, we extend final_info_gain with a column of zeros to match the new shape.
    extended_info_gain_mask = torch.cat([info_gain_mask, torch.zeros((info_gain_mask.shape[0], 1), dtype=torch.bool, device=info_gain_mask.device)], dim=1)
    extended_final_info_gain = torch.cat([final_info_gain, torch.zeros((final_info_gain.shape[0], 1), dtype=final_info_gain.dtype, device=final_info_gain.device)], dim=1)
    final_scores_per_turn = insert_at_first_false(final_scores, extended_final_info_gain, extended_info_gain_mask)

    # Apply discounted future rewards
    discounted_scores_per_turn = discounted_returns(final_scores_per_turn, gamma)

    # Locate range of indices corresponding to each response
    # IMPORANT NOTE: response_mask starts at the first response token, so the first segment corresponds to the
    # first turn's response. It also ends with padding (unless the response fills the entire length)
    # If info_gain_mask is all False, then response_mask.diff(dim=1) has one nonzero entry (corresponding to
    # the end of the first turn's response). 
    # If info_gain_mask is [True, False], then there are three nonzero entries (end of first turn's response,
    # start of second turn's response, end of second turn's response)
    # Then each additional True in info_gain_mask adds another pair of nonzero entries (start and end of another
    # turn's response)
    mask_change_locs = response_mask.diff(dim=1)#.nonzero(as_tuple=False)

    change = torch.cat([
            # first element always starts a segment
            torch.tensor([True]*mask_change_locs.shape[0], device=response_mask.device).unsqueeze(-1),  
            mask_change_locs != 0,
        ], 
        dim=1)
    segment_ids = change.cumsum(dim=1) - 1  # Shape: (bs, response_length)
    
    # Every segment with an even ID corresponds to a response segment. We insert a zero at every odd index
    # in final_scores to align the true rewards with segment_ids
    expanded_final_scores = torch.zeros((discounted_scores_per_turn.shape[0], discounted_scores_per_turn.shape[1]*2), device=final_scores.device)
    expanded_final_scores[:, 0::2] = discounted_scores_per_turn
    advantages = expanded_final_scores.gather(1, segment_ids)
    
    return advantages, advantages

def compute_advantage_igpo(
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
    
    assert adv_estimator == AdvantageEstimator.GRPO, f"adv_estimator must be GRPO for IGPO, got {adv_estimator}"
    # Initialize the mask for GRPO calculation
    grpo_calculation_mask = data.batch["response_mask"]

    advantages, returns = compute_igpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        info_gain_rewards=data.batch["info_gain"],
        info_gain_mask=data.batch["info_gain_mask"],
        response_mask=grpo_calculation_mask,
        index=data.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        gamma=gamma,
        config=config,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns

    return data

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
                    batch_size = len(batch_dict['reward_model'])
                    # Depending on the task, the pseudo-response with ground-truth answer can have a different format. For
                    # example, for math problems, the ground-truth answer needs to be put inside \boxed{...}. A config option
                    # is used to indicate what type of pseudo-response to use. The default is the one used for search tasks
                    # which matches what the IGPO paper uses.
                    task_type = self.config.algorithm.get("task_type", "search")
                    if any([batch_dict['reward_model'][i]['style'] != 'rule' for i in range(batch_size)]):
                        print("GOT STYLE OTHER THAN \'RULE\'")
                        raise ValueError("Unsupported style found in batch_dict['reward_model']")
                    # The ground-truth answers for batch i are stored in batch_dict['reward_model'][i]['ground_truth']['target']
                    # This returns a list with potentially multiple valid answers
                    # TODO: Handle multiple valid answers per sample. For now, I arbitrarily take the first one
                    # How to extract the ground-truth answers also depends on the task type
                    if task_type == "math":
                        # In this case, there is only ever one ground-truth answer per sample
                        ground_truths = [batch_dict['reward_model'][i]['ground_truth'] for i in range(batch_size)]
                    elif task_type == "search":
                        ground_truths = [batch_dict['reward_model'][i]['ground_truth']['target'] for i in range(batch_size)]
                        # if not all(len(gt)==1 for gt in ground_truths): # For checking instances with multiple valid answers
                        #     breakpoint()
                        ground_truths = [gt[0] for gt in ground_truths]  # Extract the first ground-truth answer per sample
                    else:
                        raise ValueError(f"Unsupported task_type {task_type} for IGPO trainer.")
                    # For search tasks, I use a string for the pseudo-response which is similar to what the IGPO paper uses;
                    # however, I changed the formatting a bit to match what the Qwen-2.5-1.5B-Instruct model outputs normally
                    if task_type == "search":
                        pseudo_response_template = "<think> Now there's enough information to answer.</think>\n<answer>\n{ground_truth}\n</answer><|im_end|>"
                        len_st = len(self.tokenizer("<think> Now there's enough information to answer.</think>\n<answer>\n", return_tensors="pt")['input_ids'].tolist()[0])
                        len_ed = len(self.tokenizer("\n</answer><|im_end|>", return_tensors="pt")['input_ids'].tolist()[0])
                        # The characters which will appear before and after the ground-truth answer if no merges occur
                        surrounding_chars = ['\n', '\n']
                    elif task_type == "math":
                        pseudo_response_template = "The answer is \\boxed{{{ground_truth}}}<|im_end|>"
                        len_st = len(self.tokenizer("The answer is \\boxed{", return_tensors="pt")['input_ids'].tolist()[0])
                        len_ed = len(self.tokenizer("}<|im_end|>", return_tensors="pt")['input_ids'].tolist()[0])
                        # For Qwen, '{' has ID 90 and '}' has ID 92
                        surrounding_chars = ['{', '}']
                    expected_start_token_id = self.tokenizer(surrounding_chars[0], return_tensors="pt")['input_ids'].tolist()[0][0]
                    expected_end_token_id = self.tokenizer(surrounding_chars[1], return_tensors="pt")['input_ids'].tolist()[0][0]
                    pseudo_resps_with_gt = [self.tokenizer(pseudo_response_template.format(ground_truth=ground_truth), return_tensors="pt")['input_ids'] for ground_truth in ground_truths]
                    gt_end_indices = [] # Stores the index corresponding to the last ground-truth token for each sample in the batch

                    # Pad pseudo_resps_with_gt to max_len and stack; also expand to match group size
                    max_len = max([resp_with_gt.size(1) for resp_with_gt in pseudo_resps_with_gt])
                    pad_id = self.tokenizer.pad_token_id
                    padded_resps = [
                        torch.nn.functional.pad(resp_with_gt, (0, max_len - resp_with_gt.size(1)), value=pad_id).repeat(self.config.actor_rollout_ref.rollout.n, 1)
                        for resp_with_gt in pseudo_resps_with_gt
                    ]
                    pseudo_resps_with_gt_stacked = torch.cat(padded_resps, dim=0) # Shape: (B, max_len)

                    # In certain cases, the last token of the ground-truth answer merges with the newline token that 
                    # follows it (this may happen to the start token as well, but I have not observed it yet). We can tell 
                    # if a merge happens by checking whether the newline token remains separate after tokenization. If there
                    # is a merge, we adjust the end index so that the merged token is included in the ground-truth answer span.
                    for i, resp_with_gt in enumerate(pseudo_resps_with_gt):
                        len_gt_no_markers = len(self.tokenizer(ground_truths[i], return_tensors="pt")['input_ids'].tolist()[0])
                        tokenized_gt_with_markers = self.tokenizer(f"{surrounding_chars[0]}{ground_truths[i]}{surrounding_chars[1]}", return_tensors="pt")['input_ids'].tolist()[0]
                        merge_detected = False
                        true_len_ed = len_ed
                        if tokenized_gt_with_markers[0] != expected_start_token_id:
                            true_len_ed -= 1
                            merge_detected = True
                        if tokenized_gt_with_markers[-1] != expected_end_token_id:
                            true_len_ed -= 1
                            merge_detected = True
                        if not merge_detected:
                            assert len_gt_no_markers == resp_with_gt.size(1) - len_st - len_ed, "Length mismatch even though no merges detected."
                        # The end index will account for padding
                        gt_end_indices.append(resp_with_gt.shape[1] - true_len_ed - 1)
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
                    # Also note that the number of turns for each rollout is stored in gen_batch_output.non_tensor_batch['__num_turns__']
                    with marked_timer("gt_probs", timing_raw, color="blue"):
                        turn_start_seq = torch.tensor([151644, 77091, 198]).to(gen_batch_output.batch['input_ids'].device)
                        max_num_turns = gen_batch_output.non_tensor_batch['__num_turns__'].max().item()
                        per_turn_gt_probs = -1*torch.ones((gen_batch_output.batch['input_ids'].shape[0], max_num_turns), device=gen_batch_output.batch['input_ids'].device)
                        two_turn_sequences = (gen_batch_output.non_tensor_batch['__num_turns__'] >= 2)
                        three_turn_sequences = (gen_batch_output.non_tensor_batch['__num_turns__'] >= 3)
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
                        found = matches.any(dim=1) # Will exclude any sequences where the turn start pattern was not found
                        match_locs = matches.nonzero(as_tuple=False)
                        # Convert to list of locations for each row (could be more efficient)
                        groups = {}
                        for k in match_locs[:, 0].unique():
                            groups[int(k)] = match_locs[match_locs[:, 0] == k, 1]
                        # NOTE: I get at least 2 matches for every sequence even though some have only 1 turn
                        # This is because the rollouts with 1 turn have empty user and assistant turns at the end which must be ignored
                        # In these cases, the turn_start_seq will be immediately followed by padding tokens
                        for turn_idx in range(max_num_turns):
                            active_rows = (found & (gen_batch_output.non_tensor_batch['__num_turns__'] > turn_idx)).bool()
                            # NOTE: I have found one problem in the MATH dataset where something goes wrong and the 
                            # 'assistant' token never gets added before the first turn, so turn_start_seq shows up
                            # fewer times than the number of turns given by gen_batch_output.non_tensor_batch['__num_turns__'].
                            # In such cases, I will not try to apply IGPO, so all elements of groups affected by this 
                            # will be removed from active_rows here
                            problem_indices = torch.ones(B, dtype=torch.bool, device=x.device).scatter_(0, match_locs[:, 0], False).nonzero(as_tuple=True)[0].tolist()
                            to_remove = []
                            for idx in problem_indices:
                                if idx not in to_remove:
                                    to_remove.extend(get_group_indices(B, self.config.actor_rollout_ref.rollout.n, idx))
                            to_remove = torch.tensor(to_remove, device=x.device)
                            if to_remove.numel() > 0:
                                active_rows[to_remove] = False
                            active_x = x[active_rows]
                            num_active = active_rows.sum().item()
                            ### Truncate so that we can add the ground-truth answer at the end
                            # For each active row, find the location of the (turn_idx+1)-th occurrence of turn_start_seq
                            lengths = torch.tensor([groups[i][turn_idx].item() + L for i in range(B) if active_rows[i]])
                            # We need a single max length for the output tensor
                            max_len = lengths.max().item()
                            # Build mask of "valid positions" for each row
                            # mask[b, t] = True if t < lengths[b]
                            idx = torch.arange(max_len, device=x.device)       # (max_len,)
                            mask = idx.unsqueeze(0) < lengths.unsqueeze(1)     # (B, max_len)
                            # Initialize output with padding
                            out = torch.full((num_active, max_len), pad_id,
                                            dtype=x.dtype, device=x.device)
                            
                            for i, row_idx in enumerate(active_rows.squeeze().nonzero(as_tuple=False)):
                                # r = row_idx.item()
                                # valid_len = lengths[(lengths != 0) & (active_rows)].tolist().index(lengths[r].item())
                                out[i, max_len-lengths[i]:] = active_x[i, :lengths[i]]

                            ### Add the ground-truth answers at the end of each sequence
                            pseudo_resps_with_gt_active = pseudo_resps_with_gt_stacked[active_rows].to(out.device)
                            pseudo_rollouts = torch.cat([out, pseudo_resps_with_gt_active], dim=1)
                            
                            ### Get the logprobs of the ground-truth answers
                            # Naively form the attention mask and position_ids (position_ids for each sequence will not be affected by left-padding)
                            attention_mask = (pseudo_rollouts != pad_id).long()
                            position_ids = torch.cumsum(attention_mask, dim=1) - 1
                            # Only the length of the response mask matters here; if the length is L, then only the
                            # logprobs for the last L tokens are returned
                            responses = torch.ones((pseudo_rollouts.shape[0], pseudo_resps_with_gt_active.shape[1]), dtype=torch.long)
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
                            
                            chunk_size = max(
                                1,
                                self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                            )
                            remainder = pseudo_rollout_tensordict["input_ids"].shape[0] % chunk_size
                            pad_rows = chunk_size - remainder if remainder != 0 else 0

                            if pad_rows:
                                td_device = (
                                    pseudo_rollout_tensordict.device
                                    if pseudo_rollout_tensordict.device is not None
                                    else pseudo_rollout_tensordict["input_ids"].device
                                )
                                pad_tensors = {
                                    "input_ids": torch.full(
                                        (pad_rows, pseudo_rollout_tensordict["input_ids"].shape[1]),
                                        pad_id,
                                        dtype=pseudo_rollout_tensordict["input_ids"].dtype,
                                        device=td_device,
                                    ),
                                    "attention_mask": torch.ones(
                                        (pad_rows, pseudo_rollout_tensordict["attention_mask"].shape[1]),
                                        dtype=pseudo_rollout_tensordict["attention_mask"].dtype,
                                        device=td_device,
                                    ),
                                    "position_ids": torch.zeros(
                                        (pad_rows, pseudo_rollout_tensordict["position_ids"].shape[1]),
                                        dtype=pseudo_rollout_tensordict["position_ids"].dtype,
                                        device=td_device,
                                    ),
                                    "responses": torch.zeros(
                                        (pad_rows, pseudo_rollout_tensordict["responses"].shape[1]),
                                        dtype=pseudo_rollout_tensordict["responses"].dtype,
                                        device=td_device,
                                    ),
                                }
                                pad_tensordict = TensorDict(
                                    pad_tensors,
                                    batch_size=(pad_rows,),
                                    device=td_device,
                                )
                                pseudo_rollout_tensordict = torch.cat(
                                    [pseudo_rollout_tensordict, pad_tensordict],
                                    dim=0,
                                )

                            pseudo_rollout_DP = DataProto.from_tensordict(
                                pseudo_rollout_tensordict, meta_info=gen_batch_output.meta_info
                            )

                            pseudo_log_probs = self.actor_rollout_wg.compute_log_prob(pseudo_rollout_DP).batch[
                                "old_log_probs"
                            ]
                            if pad_rows:
                                pseudo_log_probs = pseudo_log_probs[:-pad_rows]
                            gt_log_probs = pseudo_log_probs.clone()
                            gt_log_probs[:, :len_st] = 0.0  # Zero out logprobs before the start of the ground-truth answer
                            mask = torch.arange(gt_log_probs.shape[1]).unsqueeze(0) > gt_end_indices[active_rows].unsqueeze(1)
                            gt_log_probs = gt_log_probs.masked_fill(mask, 0.0)  # Zero out logprobs beyond the ground-truth answer
                            gt_probs = torch.exp(gt_log_probs.sum(dim=1))  # Sum logprobs to get total prob for the ground-truth answer
                            per_turn_gt_probs[active_rows, turn_idx] = gt_probs
                            # Log the mean of gt_probs; do not try to log per_turn_gt_probs since unused entries are -1
                            logger.log(
                                data={f"igpo/turn_{turn_idx+1}_gt_prob_mean": gt_probs.mean().detach().item()},
                                step=self.global_steps,
                            )
                        # torch.diff computes input[i + 1] - input[i]
                        info_gain = torch.diff(per_turn_gt_probs, dim=1)  # Shape: (B, max_num_turns-1)
                        info_gain_mask = (per_turn_gt_probs[:, 1:] >= 0)  # Mask indicating valid info gain entries
                        # Here we average over all sequences with at least 2 or 3 turns respectively
                        if two_turn_sequences.any():
                            logger.log(
                                data={f"igpo/turn_1_info_gain": info_gain[two_turn_sequences,0].mean().detach().item()},
                                step=self.global_steps,
                            )
                        if three_turn_sequences.any():
                            logger.log(
                                data={f"igpo/turn_2_info_gain": info_gain[three_turn_sequences,1].mean().detach().item()},
                                step=self.global_steps,
                            )
                        gen_batch_output.batch["info_gain"] = info_gain
                        gen_batch_output.batch["info_gain_mask"] = info_gain_mask

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

                        batch = compute_advantage_igpo(
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