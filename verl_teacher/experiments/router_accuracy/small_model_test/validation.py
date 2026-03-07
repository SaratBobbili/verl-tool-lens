""" Much of this code was adopted from test_special_dp_teacher_score.py """

import os, argparse, json

import torch
import torch.distributed
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoConfig

from verl import DataProto
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.engine import FSDPEngineConfig

from verl_teacher.workers.config import FSDPTeacherConfig, FSDPTeacherModelCfg
from verl_teacher.workers.fsdp_workers import TeacherScoreWorker

# Usage for this experiment: python validation.py validation_data/<filename> teacher_final_ckpt/huggingface
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file_path", type=str, help="Path to the validation data file (JSONL format)")
    ap.add_argument("model_path", type=str, help="Path to the teacher model")
    args = ap.parse_args()

    """Set up distributed environment"""
    if not torch.distributed.is_initialized():
        # If environment variables for distributed setup are not set, we can initialize a default process group for testing
        # Only overwrite those which are not already set
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://"
        )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path.rstrip("/")  # Remove trailing slash if present
    # config = AutoConfig.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    config = FSDPTeacherConfig(
        strategy="fsdp2",
        mini_batch_size=128,
        micro_batch_size_per_gpu=32,
        forward_micro_batch_size_per_gpu=32,
        epochs=1,
        cliprange_value=0.5,
        grad_clip=1.0,
        use_dynamic_bsz=False,
        ulysses_sequence_parallel_size=1,
        rollout_n=1,
        optim=FSDPOptimizerConfig(lr=1e-6),
        model=FSDPTeacherModelCfg(
            path=model_path,
            tokenizer_path=model_path,
            fsdp_config=FSDPEngineConfig(fsdp_size=-1),
            use_remove_padding=False,
            use_mean_pooling=False,
        ),
    )

    tokenizer = hf_tokenizer(model_path, trust_remote_code=False)

    worker = TeacherScoreWorker(config)
    worker.init_model()

    """ Get validation data from files """
    assert os.path.exists(args.file_path), f"Validation file {args.file_path} does not exist"
    with open(args.file_path, "r") as f:
        data_raw = [json.loads(line) for line in f.readlines()]
    prompts = [item["input"] for item in data_raw]
    scores = [item["avg_score"] for item in data_raw]  # Assuming avg_score is already in [0, 1]
    tokenized = tokenizer(prompts, padding=True, padding_side='right', return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    scores = torch.tensor(scores).unsqueeze(1)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=[input_ids.shape[0]],
    )
    data = DataProto(batch=data)

    result = worker.compute_scores(data)
    pred_scores = result.batch["scores"]
    error = torch.abs(pred_scores - scores)
    mae = torch.mean(error).item()
    print(f"Mean Absolute Error on validation set: {mae:.4f}")
    breakpoint()  # For inspection of pred_scores vs. true scores

    """Clean up distributed environment"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

            