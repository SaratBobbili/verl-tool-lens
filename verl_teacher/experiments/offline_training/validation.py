import os
import argparse
import json

import torch
import torch.distributed
import matplotlib.pyplot as plt

from tensordict import TensorDict

from verl import DataProto
from verl.utils.tokenizer import hf_tokenizer
from verl.workers.config import FSDPOptimizerConfig
from verl.workers.config.engine import FSDPEngineConfig

from verl_teacher.workers.config import FSDPTeacherConfig, FSDPTeacherModelCfg
from verl_teacher.workers.fsdp_workers import TeacherScoreWorker


def compute_routing_metrics(pred_scores: torch.Tensor, scores: torch.Tensor, threshold: float):
    """
    Compute routing metrics for a single threshold.

    Args:
        pred_scores: Tensor of predicted scores, shape [N] or [N, 1]
        scores: Tensor of true binary scores (0 or 1), shape [N] or [N, 1]
        threshold: Routing threshold

    Returns:
        dict with:
            - percent_accepted
            - percent_correct_accepted
            - percent_correct_non_accepted
    """
    pred_scores = pred_scores.reshape(-1)
    scores = scores.reshape(-1)

    accepted = pred_scores > threshold
    non_accepted = ~accepted

    num_total = accepted.numel()
    num_accepted = accepted.sum().item()
    num_non_accepted = non_accepted.sum().item()

    percent_accepted = 100.0 * num_accepted / num_total if num_total > 0 else 0.0

    correct_accepted = ((accepted) & (scores == 1.0)).sum().item()
    percent_correct_accepted = (
        100.0 * correct_accepted / num_accepted if num_accepted > 0 else 100.0
    )

    correct_non_accepted = ((non_accepted) & (scores == 1.0)).sum().item()
    percent_correct_non_accepted = (
        100.0 * correct_non_accepted / num_non_accepted if num_non_accepted > 0 else 0.0
    )

    return {
        "percent_accepted": percent_accepted,
        "percent_correct_accepted": percent_correct_accepted,
        "percent_correct_non_accepted": percent_correct_non_accepted,
    }


# Usage for this experiment:
# python validation.py validation_data/<filename> teacher_final_ckpt/huggingface
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file_path", type=str, help="Path to the validation data file (JSONL format)")
    ap.add_argument("model_path", type=str, help="Path to the teacher model")
    ap.add_argument(
        "--routing_threshold",
        type=float,
        default=None,
        help=(
            "Threshold for routing decisions. "
            "If >= 0, evaluate that single threshold. "
            "If < 0, evaluate a list of thresholds and plot metrics."
        ),
    )
    ap.add_argument(
        "--threshold_list",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="List of routing thresholds to evaluate when routing_threshold < 0",
    )
    ap.add_argument(
        "--plot_path",
        type=str,
        default="routing_threshold_plot.png",
        help="Where to save the threshold sweep plot when routing_threshold < 0",
    )
    args = ap.parse_args()

    # Set up distributed environment
    if not torch.distributed.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )

    rank = torch.distributed.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    model_path = args.model_path.rstrip("/")

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

    # Get validation data from file
    assert os.path.exists(args.file_path), f"Validation file {args.file_path} does not exist"
    with open(args.file_path, "r") as f:
        data_raw = [json.loads(line) for line in f.readlines()]

    prompts = [item["input"] for item in data_raw]
    scores = [item["avg_score"] for item in data_raw]  # assumed to already be in [0, 1]

    tokenized = tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

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
    pred_scores = result.batch["scores"].detach().cpu()
    scores = scores.detach().cpu()

    if args.routing_threshold is not None:
        assert torch.logical_or(scores == 0.0, scores == 1.0).all(), (
            "Routing threshold evaluation requires binary scores (0 or 1)"
        )

        if args.routing_threshold < 0:
            thresholds = args.threshold_list

            percent_accepted_list = []
            percent_correct_accepted_list = []
            percent_correct_non_accepted_list = []

            print("Evaluating routing metrics across thresholds:")
            for threshold in thresholds:
                metrics = compute_routing_metrics(pred_scores, scores, threshold)
                percent_accepted_list.append(metrics["percent_accepted"])
                percent_correct_accepted_list.append(metrics["percent_correct_accepted"])
                percent_correct_non_accepted_list.append(metrics["percent_correct_non_accepted"])

                print(
                    f"Threshold={threshold:.3f} | "
                    f"accepted={metrics['percent_accepted']:.2f}% | "
                    f"correct among accepted={metrics['percent_correct_accepted']:.2f}% | "
                    f"correct among non-accepted={metrics['percent_correct_non_accepted']:.2f}%"
                )

            plt.figure(figsize=(8, 5))
            plt.plot(thresholds, percent_accepted_list, marker="o", label="% of answers accepted")
            plt.plot(
                thresholds,
                percent_correct_accepted_list,
                marker="o",
                label="% of correct answers among accepted",
            )
            plt.plot(
                thresholds,
                percent_correct_non_accepted_list,
                marker="o",
                label="% of correct answers among non-accepted",
            )
            plt.xlabel("Routing threshold (predicted success probability must exceed this for acceptance)")
            plt.ylabel("Percentage")
            plt.title(f"Routing metrics vs. threshold - {args.file_path.split('/')[-1]}")
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.plot_path, dpi=200)
            print(f"Saved threshold sweep plot to: {args.plot_path}")

        else:
            metrics = compute_routing_metrics(pred_scores, scores, args.routing_threshold)
            print(
                f"With a routing threshold of {args.routing_threshold}, "
                f"{metrics['percent_accepted']:.2f}% of samples would be accepted."
            )
            print(
                f"Of the accepted samples, "
                f"{metrics['percent_correct_accepted']:.2f}% were actually correct."
            )
            print(
                f"Of the non-accepted samples, "
                f"{metrics['percent_correct_non_accepted']:.2f}% were actually correct."
            )

    else:
        error = torch.abs(pred_scores.reshape(-1) - scores.reshape(-1))
        mae = torch.mean(error).item()
        print(f"Mean Absolute Error on validation set: {mae:.4f}")

    # Clean up distributed environment
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()