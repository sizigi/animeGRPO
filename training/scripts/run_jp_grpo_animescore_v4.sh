#!/bin/bash
set -x

# Phase 4: Soft CER penalty (no hard -1.0 cutoff)
# - R = 0.6 * animescore + 0.4 * max(0, 1 - 2*CER)
# - CER>0.5 still gets animescore gradient (not killed to -1.0)
# - KL coef=0.05 (proven stable in v3)
# - Speech-only token masking
# - Resume from step 420 checkpoint

export ANIMESCORE_SERVER=http://localhost:8002
export WHISPER_SERVER=http://localhost:8001
export RAY_DISABLE_MEMORY_MONITOR=1

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.05 \
    actor_rollout_ref.actor.clip_ratio=0.1 \
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
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.speech_only_generation=true \
    +actor_rollout_ref.rollout.speech_token_start_id=128264 \
    +actor_rollout_ref.rollout.speech_token_end_id=193799 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    custom_reward_function.path=verl/utils/reward_score/tts_animescore_cer.py \
    custom_reward_function.name=compute_score \
    trainer.project_name='llasa_tts_grpo_jp_animescore_v4' \
    trainer.experiment_name='soft_cer_speech_mask' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=10 "$@"
