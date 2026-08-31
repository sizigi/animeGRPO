#!/bin/bash
set -x

# v5-likability: 1B-Multilingual + Likability objective + CER hard-constraint + adaptive KL
# ----------------------------------------------------------------------------
# Ablation of run_jp_grpo_utmos_constrained_v5.sh — only difference is
# UTMOS swapped for Likability (CocoNut-Humoresque WavLM-Base+ predictor)
# as the reward axis. All other knobs identical.
# ----------------------------------------------------------------------------

export LIKABILITY_SERVER=http://localhost:8003
export WHISPER_SERVER=http://localhost:8001
export RAY_DISABLE_MEMORY_MONITOR=1

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.type=adaptive \
    algorithm.kl_ctrl.kl_coef=0.05 \
    algorithm.kl_ctrl.target_kl=0.05 \
    algorithm.kl_ctrl.horizon=2000 \
    data.train_files=$HOME/data/llasa-tts-rl-grpo-jp-v2/train.parquet \
    data.val_files=$HOME/data/llasa-tts-rl-grpo-jp-v2/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.truncation='error' \
    actor_rollout_ref.model.path=HKUSTAudio/Llasa-1B-Multilingual \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio=0.1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    '+actor_rollout_ref.rollout.stop_token_ids=[128261]' \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.85 \
    '+actor_rollout_ref.rollout.repetition_penalty=1.05' \
    actor_rollout_ref.rollout.n=4 \
    +actor_rollout_ref.rollout.speech_only_generation=true \
    +actor_rollout_ref.rollout.speech_token_start_id=128264 \
    +actor_rollout_ref.rollout.speech_token_end_id=193799 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.85 \
    custom_reward_function.path=verl/utils/reward_score/tts_likability_cer_constraint.py \
    custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    trainer.project_name='llasa_1b_grpo_v5_likability_constrained' \
    trainer.experiment_name='likability_cer_constraint_adaptive_kl' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=40 "$@"
