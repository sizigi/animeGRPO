# Training runbook

Everything here targets one setup: `HKUSTAudio/Llasa-1B-Multilingual` +
GRPO + a perceptual reward gated on CER, run with the
[channel-io/ch-tts-llasa-rl-grpo](https://github.com/channel-io/ch-tts-llasa-rl-grpo)
verl fork.

## Hardware

The paper's 1B runs fit on **two GPUs**, but not in the way you might expect:
GPU 0 holds the trainer (policy + reference + vLLM rollout), GPU 1 holds the
reward servers. Whisper large-v3 plus AnimeScore is roughly 16.5 GB, so the
scorers can share a card with anything small — but they cannot share a card
with the trainer at `gpu_memory_utilization=0.6`.

Step time is ~10–15 s at `train_batch_size=16, n=4`; the paper's 1400-step run
takes about 12 hours on an H100.

## 1. Trainer setup

```bash
git clone https://github.com/channel-io/ch-tts-llasa-rl-grpo
cd ch-tts-llasa-rl-grpo
git apply /path/to/animeGRPO/training/verl.patch
pip install -e .
cp /path/to/animeGRPO/training/reward_functions/tts_*.py verl/utils/reward_score/
```

`verl.patch` is 71 lines across 7 files. The two that matter:

- **`vllm_rollout*.py`** adds `speech_only_generation`. Rollouts are constrained
  to the codec-token range `[128264, 193799]` plus the stop token, so the policy
  cannot emit text tokens into a speech sequence. Without this, a drifting
  policy produces sequences the codec cannot decode and every downstream score
  is garbage. It also coerces `stop_token_ids` from OmegaConf to a plain list,
  which vLLM requires.
- **`ray_trainer.py`** allows a reward function to return ragged per-token
  weights (`token_weights`) and applies them to the advantages — used by the
  token-selective ablation, harmless for the main runs.

The rest are micro-batch and FSDP-config fixes.

## 2. Reward servers

```bash
REWARD_GPU=1 ANIMESCORE_CKPT=/path/to/ckpt_wavlm_frozen_best.pt \
  bash scripts/start_reward_servers.sh
```

Defaults to Whisper (:8001) + AnimeScore (:8002); set
`AXES="animescore utmos likability vad"` for the other axes. Details and the
REST contract: [reward_servers/README.md](reward_servers/README.md).

Health-check before launching a run — a reward function that cannot reach its
server raises on the first batch:

```bash
for p in 8001 8002; do curl -s localhost:$p/healthz; echo " :$p"; done
```

## 3. Data

```bash
python data_prep/prepare_jp_data_v2.py     # → ~/data/llasa-tts-rl-grpo-jp-v2/{train,test}.parquet
python data_prep/build_train_en.py         # EN train
python data_prep/build_test_v2_en.py       # EN test/val
```

JP training uses 900 prompts / 100 val; EN uses 900 / 50. The released prompt
CSVs in `../data/prompts/` are the same sets, exported.

> The EN run used the held-out test set as its validation set — a limitation
> noted in the paper's appendix. It is preserved here rather than quietly fixed,
> so the released configs match what was actually run.

## 4. Launch

```bash
bash scripts/run_jp_grpo_animescore_constrained_v5.sh     # the main result
```

One script per paper run:

| Script | What it trains |
|---|---|
| `run_jp_grpo_animescore_constrained_v5.sh` | **AnimeScore + zone-CER, JP** — main |
| `run_en_1b_grpo_animescore_constrained_v5.sh` | AnimeScore + zone-CER, EN |
| `run_jp_grpo_animescore_v4.sh` | scalar-CER weighted sum (ablation) |
| `run_jp_grpo_animescore_v6_AS_only.sh` | AnimeScore alone, no CER (ablation) |
| `run_jp_grpo_utmos_constrained_v5.sh` | UTMOS + zone-CER |
| `run_jp_grpo_likability_constrained_v5.sh` | Likability + zone-CER |
| `run_jp_grpo_vad_arousal_constrained_v5.sh` | VAD arousal + zone-CER (negative result) |

All of them are `verl.trainer.main_ppo` invocations with
`algorithm.adv_estimator=grpo`; the axis is selected by
`custom_reward_function.path`. Hyperparameters are mirrored as YAML in
[configs/](configs) for reading without parsing a shell script.

The settings that are load-bearing rather than arbitrary:

| Setting | Value | Why |
|---|---|---|
| `rollout.top_p` | 0.85 | 0.95 let the JP policy drift off-language |
| `rollout.repetition_penalty` | 1.05 | suppresses the codec-token stutter loop |
| `rollout.n` | 4 | GRPO group size; the advantage baseline is this group |
| `actor.clip_ratio` | 0.1 | 0.2 was unstable on a 1B policy at this LR |
| `actor.use_kl_loss` | **False** | KL is applied in-reward; both would double-count |
| `algorithm.kl_ctrl.type` | adaptive | target 0.05, horizon 2000 |
| `max_actor_ckpt_to_keep` | 3 | ⚠ see below |

> ⚠ `max_actor_ckpt_to_keep=3` silently deletes older checkpoints. One paper run
> (likability) lost its true-best step this way and reports the best *surviving*
> checkpoint instead. If you care about post-hoc step selection, raise this or
> copy promising checkpoints aside as they appear.

## 5. Checkpoint selection

Do **not** select on the raw axis score. In the main run, raw validation
AnimeScore keeps climbing to step 1710 while the violation rate climbs with it
(0.26 → 0.31). The constraint-aware validation reward peaks at **1400** and
declines afterward — that is the signal to select on, and it is the whole point
of the gate. Numbers in [../RESULTS.md](../RESULTS.md#5-kl-dynamics-and-checkpoint-selection).

## 6. KL re-anchoring (optional)

If KL drift accumulates past the target for long stretches, re-anchor the
reference model:

1. Stop at a stable checkpoint `S`.
2. `cp -r checkpoints/<exp>/global_step_S checkpoints/<exp>/ref_anchor_S`
3. Set `actor_rollout_ref.model.path=checkpoints/<exp>/ref_anchor_S`.
4. Resume — the latest actor is loaded as init, and the reference is now fresh.

None of the paper's runs required this; the adaptive controller held the
realized KL penalty at 0.041 against a 0.05 target on its own.
