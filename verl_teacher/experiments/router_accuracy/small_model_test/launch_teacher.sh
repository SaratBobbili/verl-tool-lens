project_name="verl_teacher_experiments"
run_name="router_accuracy_small_model_test"
n_nodes=1
n_gpus_per_node=4

PYTHONUNBUFFERED=1 python3 -m verl_teacher.main_teacher \
    teacher.model.use_mean_pooling=False \
    teacher.use_mse_loss=True \
    teacher.model.path="Qwen/Qwen2.5-Math-1.5B" \
    teacher.micro_batch_size_per_gpu=32 \
    teacher.mini_batch_size=128 \
    teacher.optim.lr=1e-6 \
    teacher.checkpoint.save_contents=['model','extra','hf_model'] \
    data_source=offline \
    offline_data_path="./training_data/" \
    data.train_batch_size=128 \
    data.train_files=$train_data \
    data.val_files=$val_data \
    trainer.project_name=$project_name \
    trainer.logger=['console'] \
    trainer.experiment_name=$run_name \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.default_local_dir="." \
    trainer.total_epochs=1 \
    2>&1 | tee tmp_teacher.log