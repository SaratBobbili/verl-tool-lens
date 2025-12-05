set -x
dataset_name=gsm8k_torl # or math_torl_offical to use torl training data
train_data=$(pwd)/data/${dataset_name}/train.parquet
val_data=$(pwd)/data/${dataset_name}/test.parquet
model_name=Qwen/Qwen2.5-Coder-1.5B-Instruct
rl_alg=mtgrpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=1
n_nodes=1
n=8
batch_size=64
ppo_mini_batch_size=1
max_prompt_length=512
max_response_length=1536
max_obs_length=1024
temperature=1.0
top_p=1.0
enable_agent=True # enable agent for tool use
strategy="fsdp"
action_stop_tokens='```output'
max_turns=1
kl_loss_coef=0.0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=torl_mtgrpo
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=2
tensor_model_parallel_size=1
gpu_memory_utilization=0.4 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=False # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151645] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=True # enable multi-turn training
max_action_length=512
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="acc-only"
if [ -n "$CUDA_VISIBLE_DEVICES" ] || [ -n "$HIP_VISIBLE_DEVICES" ]; then
    unset ROCR_VISIBLE_DEVICES
fi

# Disable VLLM v1 for GPUs with compute capability < 8.0 (e.g., V100)
# Set to 1 only if you have GPU with compute capability >= 8.0 (e.g., A100, H100)
export VLLM_USE_V1=1
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO

rollout_mode='async'

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" --workers_per_tool 8 &
server_pid=$!

echo "Server (pid=$server_pid) started at $tool_server_url"

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','wandb'] \
    trainer.project_name="GSM8K_MULTITURN_EXPERIMENTS" \
    trainer.experiment_name="MTGRPO_3" \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=200 \
    trainer.test_freq=20 \
    trainer.total_training_steps=100

echo "Training completed!"