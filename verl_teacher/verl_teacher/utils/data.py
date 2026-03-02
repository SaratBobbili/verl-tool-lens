""" Utility classes and functions for handling data used by the teacher model """

__all__ = ["zero_pad_dataproto","extract_question_from_chat_template"]

import torch
from tensordict import TensorDict
from verl.protocol import DataProto

def zero_pad_dataproto(dp, target_bs):
    """
    Pads a DataProto to a target batch size by adding zero entries. If the DataProto's batch size is already 
    equal to the target, it is returned unchanged. If the DataProto's batch size is greater than the target, 
    an error is raised. An additional meta-info field "non_pad_indices" is added to indicate which entries are 
    original vs. padded.
    """
    cur = len(dp)
    if cur == target_bs:
        return dp
    if cur > target_bs:
        raise ValueError(f"DataProto batch size {cur} is greater than target batch size {target_bs}")

    td = dp.batch
    pad_shape = (target_bs - cur,)
    padded = {}
    for k, v in td.items():
        padded[k] = torch.cat([v, torch.zeros(pad_shape + v.shape[1:], device=v.device, dtype=v.dtype)], dim=0)

    meta_info = dp.meta_info.copy()
    meta_info["non_pad_indices"] = list(range(cur))

    return DataProto(batch=TensorDict(padded, batch_size=[target_bs]),
                     non_tensor_batch=dp.non_tensor_batch,
                     meta_info=meta_info)

def extract_question_from_chat_template(prompt: str) -> str:
    """
    Extracts the question portion from a prompt formatted in verl-tool's chat template style.
    If the prompt does not use the expected chat template, only whitespace will be stripped
    """
    if "<|im_start|>user\n" in prompt and "<|im_end|>\n<|im_start|>assistant" in prompt:
        start = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        end = prompt.index("<|im_end|>\n<|im_start|>assistant")
        return prompt[start:end].strip()
    else:
        # If the expected format is not found, return the original prompt
        return prompt.strip()