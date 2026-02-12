#!/bin/bash
#SBATCH -J hrla-7bcoder-tool-train
#SBATCH -p gpu-research
#SBATCH --qos=olympus-research-gpu
#SBATCH --gres=gpu:a100:4
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

bash examples/train/math_tir/v1_train_7b_grpo_multi-turn_w_tool.sh

