#!/bin/bash
#SBATCH -J hrl_a-eval
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH -t 4-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) activate your env/modules here
# module load cuda/12.1
# source ~/.bashrc && conda activate verl-tool-env


# model_path="/mnt/shared-scratch/Shakkottai_S/saratb/verl-tool-lens/checkpoints/creditsummary/torl-fsdp-agent-qwen_qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6acc-only-4gpu/global_step_5/actor"
# test_task="math"
# test_type="chat_completion"
# base_url="http://localhost:5000"
# python eval_service/test/test_api.py --model_name $model_path --test_task $test_task --test_type $test_type --base_url $base_url

# bash eval_service/scripts/start_api_service.sh &
bash benchmarks/math-evaluation-harness/scripts/v1_run_eval_math_greedy_deepmath.sh
