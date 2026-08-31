# animeGRPO

**Constrained perceptual GRPO for codec-based speech language models.**

Companion code for *"When Does Predictor-Based RL Align with Human Perception?
A Study of Subjective Rewards in Codec-Based Speech Language Models"*
(Joonyong Park and Jerry Li, Spellbrush — arXiv preprint, 2026).

🔊 **[Audio demo →](https://sizigi.github.io/animeGRPO/)**

This repository shows how to post-train a [LLaSA](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual)
codec speech LM with GRPO against **AnimeScore**, a preference-trained
anime-likeness reward — and how to keep the policy from destroying
intelligibility while it chases that reward.

---

## The problem, in one figure's worth of words

A perceptual reward like AnimeScore is *subjective*: nothing in it says the
utterance has to remain a valid reading of the prompt. Optimize it directly and
the policy discovers that garbled, over-acted noise scores well — in our runs
**34% of outputs collapsed** (post-retry CER > 0.30) while AnimeScore went up.

Mixing CER into the reward as a weighted sum only halves the problem: it creates
a Pareto trade-off the policy can pay its way out of, sacrificing intelligibility
for style. Violations drop 34% → 24%, but half the style gain goes with it.

The fix used here is a **zone-based hard constraint**. CER is a *gate*, not a term:

```
                         R = max(0, AnimeScore + 3.0)
    CER ≤ 0.10   CLEAN      + 0.5                      ← small bonus
    CER ≤ 0.30   FEASIBLE   + 0.0                      ← optimize style here
    CER >  0.30  VIOLATE    R = −1.0                   ← style gradient suppressed
```

The ordering invariant `violate < feasible ≤ clean` holds for every AnimeScore
observed on this base model, so there is no "fail cheaply" mode: the policy has
to satisfy intelligibility *first*, then maximize style inside the feasible zone.
That cuts the violation rate to **6%** while keeping **64% of the AS-only style
gain**.

See [RESULTS.md](RESULTS.md) for the full numbers.

---

## Repository layout

```
training/
  reward_functions/   verl custom reward modules — the zone-CER gate lives here
  reward_servers/     FastAPI scorers (AnimeScore, Whisper/CER, UTMOS, likability, VAD)
  scripts/            one launch script per paper run
  data_prep/          prompt sets → parquet for the trainer
  configs/            hyperparameters as YAML, one per axis
  verl.patch          the 71-line change to the verl fork that makes speech rollouts work
evaluation/
  merge_1b_grpo.py    FSDP checkpoint → HuggingFace model
  evaluate_*.py       test-set generation + AS/CER/UTMOS scoring with CER retry
  paper_tables/       recompute each paper table from the released CSVs
data/
  prompts/            JP and EN train/val/test prompt sets
  machine_scores/     per-prompt and per-candidate scoring behind every table
docs/                 the audio demo page (GitHub Pages)
```

---

## How the training loop fits together

```
   prompt ──► LLaSA-1B policy ──► speech tokens ──┬──► XCodec2 ──► wav ──► AnimeScore server :8002
   (verl+vLLM rollout, speech-token-masked)       │                              │
                                                  └──► XCodec2 ──► wav ──► Whisper server :8001
                                                                                 │
                                     reward fn (zone gate) ◄──── AnimeScore, CER ┘
                                                  │
                                     GRPO advantage ──► policy update (adaptive KL to ref)
```

The reward functions are stateless HTTP clients. The scorer models live in
separate processes on their own GPU, so the trainer never has to hold them in
memory — which is what makes single-GPU 1B GRPO fit at all.

---

## Quickstart (JP AnimeScore run — the paper's main result)

**1. Get the trainer.** The runs use the
[channel-io/ch-tts-llasa-rl-grpo](https://github.com/channel-io/ch-tts-llasa-rl-grpo)
verl fork, plus a small patch for speech-token-only rollouts:

```bash
git clone https://github.com/channel-io/ch-tts-llasa-rl-grpo
cd ch-tts-llasa-rl-grpo
git apply /path/to/animeGRPO/training/verl.patch
pip install -e .
```

**2. Install the reward functions** into the trainer's reward-score directory:

```bash
cp /path/to/animeGRPO/training/reward_functions/tts_*.py verl/utils/reward_score/
```

**3. Start the reward servers** (they need their own GPU — ~16.5 GB for
Whisper large-v3 + AnimeScore):

```bash
REWARD_GPU=1 ANIMESCORE_CKPT=/path/to/animescore.pt \
  bash /path/to/animeGRPO/training/scripts/start_reward_servers.sh
# health: curl localhost:8002/healthz   (note: /healthz, not /health)
```

**4. Build the prompt parquet:**

```bash
python /path/to/animeGRPO/training/data_prep/prepare_jp_data_v2.py
# → ~/data/llasa-tts-rl-grpo-jp-v2/{train,test}.parquet
```

**5. Train:**

```bash
bash /path/to/animeGRPO/training/scripts/run_jp_grpo_animescore_constrained_v5.sh
```

**6. Merge and evaluate** the selected checkpoint:

```bash
python evaluation/merge_1b_grpo.py \
    --ckpt_dir  checkpoints/<exp>/global_step_1400 \
    --output_dir ~/checkpoints/llasa-1b-v5-step1400-merged
python evaluation/evaluate_1b_v5_test_v2.py
```

Full runbook, including checkpoint selection and the KL re-anchoring
procedure: [training/README.md](training/README.md).

---

## Models and data you need to bring

| Component | Source | Redistributed here |
|---|---|---|
| Policy base | [HKUSTAudio/Llasa-1B-Multilingual](https://huggingface.co/HKUSTAudio/Llasa-1B-Multilingual) | no |
| Codec | [HKUST-Audio/xcodec2](https://huggingface.co/HKUST-Audio/xcodec2) | no |
| AnimeScore reward | [sizigi/animescore](https://github.com/sizigi/animescore) — RankNet head on frozen WavLM-base | no (weights) |
| CER | `openai/whisper-large-v3` | no |
| UTMOS | [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS) UTMOS22-strong | no |
| VAD arousal | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | no |
| Prompt sets | this repo, `data/prompts/` | **yes** |
| Scoring outputs | this repo, `data/machine_scores/` | **yes** |
| Evaluation audio | this repo, `docs/audio/` | **yes** |

Policy checkpoints are not published: LLaSA inherits a LLaMA-family license, so
only adapters would be safely redistributable. Selected steps are recorded in
[RESULTS.md](RESULTS.md) so the runs can be reproduced.

**Not included:** the human listening-test votes and their analysis code. The
preference data is anonymized but stays out of this release; the machine-side
CSVs here reproduce every machine table on their own.

---

## Reward axes

`training/reward_functions/` carries one module per axis, all sharing the same
zone gate — the ablation in the paper is exactly "swap the axis, keep the gate":

| Module | Axis | Selected step |
|---|---|---|
| `tts_animescore_cer_constraint.py` | AnimeScore (zone-CER) — **main** | JP 1400 · EN 900 |
| `tts_utmos_cer_constraint.py` | UTMOS naturalness | 1020 |
| `tts_likability_cer_constraint.py` | Likability classifier | 540 |
| `tts_vad_arousal_cer_constraint.py` | VAD arousal — *negative result* | 240 |
| `tts_animescore_cer.py` | AnimeScore, scalar-CER weighted sum (ablation) | 480 |
| `tts_animescore.py` | AnimeScore only, no CER (ablation) | 540 |
| `cer_zone.py` | framework-agnostic reference implementation of the gate | — |

---

## Citation

```bibtex
@misc{park2026predictorrl,
  title  = {When Does Predictor-Based {RL} Align with Human Perception?
            A Study of Subjective Rewards in Codec-Based Speech Language Models},
  author = {Park, Joonyong and Li, Jerry},
  year   = {2026},
  note   = {arXiv preprint}
}
```

The arXiv identifier will be added to the entry once assigned.

## License

Code is Apache-2.0 ([LICENSE](LICENSE)), matching the verl fork it extends.

Audio under `docs/audio/` and the CSVs under `data/` are released for research
inspection and replication of the paper's results. **The generated audio must
not be used for impersonation or unauthorized style imitation of any voice, real
or fictional** — some clips were synthesized under reward objectives that push
delivery toward anime voice-acting performance.
