#!/bin/bash

# 多工具推理脚本
# 使用训练框架进行单轮推理
# pm2 start "msb server start --dev" --name msb-mcp

set -x

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True 
export TAVILY_API_KEY=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 
export WANDB_API_KEY=bb8507670acdf8e05b2059b640b4e1d268b96aae 
export GEMINI_API_KEY=AIzaSyDV2-O8ELQb2ZCFxi9gzIchr2GA60Ii8Hg

path_prefix=/workspace/verl
data_path=$path_prefix/data/eval/eval_data.parquet
save_path=$path_prefix/data/eval/Qwn2.5_3b_Instruct_sft_tool_use_gen_test.parquet
model_path=$path_prefix/saved_models/checkpoint/multiturn-sft-qwen-2.5-3b-instruct/merged_step_1536
tool_config_path=$path_prefix/verl/examples/sglang_multiturn/config/tool_config/mcp_tool_config.yaml
# tool_config_path=$path_prefix/verl/recipe/retool/sandbox_fusion_tool_config.yaml

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path \
    data.val_files=$data_path \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.prompt_key=prompt \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    custom_reward_function.path=/workspace/verl/verl/recipe/mcp_tool/mcp_tool.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='mcp_tool_framework_test_1' \
    trainer.experiment_name='mcp_test'
