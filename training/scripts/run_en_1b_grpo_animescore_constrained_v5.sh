#!/bin/bash
set -x

# v5-EN: 1B-Multilingual + EN data + AnimeScore + CER hard-constraint + adaptive KL
# ----------------------------------------------------------------------------
# English variant of run_jp_grpo_animescore_constrained_v5.sh.
# Same model (Llasa-1B-Multilingual), same reward design, same hyperparameters;
# only the dataset is swapped to llasa-tts-rl-grpo-en-v2 (Wikipedia-derived
# 900 train / 50 test_v2 EN sentences) and project/experiment names are renamed.
#
# Sanity context (from prior 3B-EN run):
#   - AnimeScore on EN base: mean -0.83, std 1.54, range [-4.43, +2.14]
#   - G2(anime) - G3(neutral) = +2.47 → AS classifier still discriminates EN
#
# Servers required (both already on GPU 1):
#   - AnimeScore @ 8002
#   - Whisper    @ 8001
# ----------------------------------------------------------------------------

export ANIMESCORE_SERVER=http://localhost:8002
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
    data.train_files=$HOME/data/llasa-tts-rl-grpo-en-v2/train.parquet \
    data.val_files=$HOME/data/llasa-tts-rl-grpo-en-v2/test_v2.parquet \
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
    custom_reward_function.path=verl/utils/reward_score/tts_animescore_cer_constraint.py \
    custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    trainer.project_name='llasa_1b_en_grpo_v5_constrained' \
    trainer.experiment_name='animescore_cer_constraint_adaptive_kl_en' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=40 "$@"
