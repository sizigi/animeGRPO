#!/bin/bash
set -x

# v5: 1B-Multilingual + AnimeScore objective + CER hard-constraint + adaptive KL
# ----------------------------------------------------------------------------
# Improvements over v4 (1+2+3 from analysis):
#   1. KL stabilization
#      - Adaptive KL controller (in-reward path) with target_kl=0.05
#      - GRPO actor KL loss disabled (use_kl_loss=False) to avoid double-count
#      - Manual reference re-anchoring: re-run with --resume + new ref ckpt
#        every ~500 steps if KL drift accumulates (see RUNBOOK below)
#
#   2. CER as hard constraint (not weighted-sum)
#      - tts_animescore_cer_constraint.py:
#          CER > 0.30 → R = -0.5  (animescore gradient suppressed)
#          CER ≤ 0.30 → R = animescore (+ bonus when CER ≤ 0.10)
#      - Pushes model to satisfy intelligibility BEFORE optimizing style
#
#   3. Sampling tightening (prevents off-language drift / repetition)
#      - top_p:               0.95 → 0.85
#      - repetition_penalty:  1.0  → 1.05
#      - speech_only_generation kept (codec-token mask, prevents text leak)
#
# RUNBOOK for reference re-anchoring (optional, if KL > target_kl persistently):
#   1. Stop training at a stable checkpoint S
#   2. Copy ckpt actor weights to a new "ref" location:
#        cp -r checkpoints/<exp>/global_step_S checkpoints/<exp>/ref_anchor_S
#   3. Set actor_rollout_ref.model.path=checkpoints/<exp>/ref_anchor_S
#      (this becomes the new ref)
#   4. Resume training (auto-loads latest actor as init, ref is fresh)
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
    custom_reward_function.path=verl/utils/reward_score/tts_animescore_cer_constraint.py \
    custom_reward_function.name=compute_score \
    reward_model.launch_reward_fn_async=True \
    trainer.project_name='llasa_1b_grpo_v5_constrained' \
    trainer.experiment_name='animescore_cer_constraint_adaptive_kl' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode='auto' \
    trainer.total_epochs=40 "$@"
