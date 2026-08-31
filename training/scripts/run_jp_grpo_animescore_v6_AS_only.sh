#!/bin/bash
set -x

# v6 AS-only: 1B-Multilingual + raw AnimeScore reward (NO CER constraint).
# All other settings IDENTICAL to v5 zone-CER (run_jp_grpo_animescore_constrained_v5.sh)
# so the only variable is the reward shaping. Apples-to-apples ablation for EMNLP.
#
# Expected behavior: AS climbs faster than v5 in early steps, then CER also climbs
# (collapse) because no CER feedback in reward. KL adaptive controller alone cannot
# prevent collapse.

export ANIMESCORE_SERVER=http://localhost:8002
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
    custom_reward_function.path=verl/utils/reward_score/tts_animescore.py \
    custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    trainer.project_name='llasa_1b_grpo_v6_AS_only' \
    trainer.experiment_name='animescore_no_cer' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=40 "$@"
