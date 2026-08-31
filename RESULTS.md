# Results

All runs: `HKUSTAudio/Llasa-1B-Multilingual`, GRPO, `n=4` rollouts per prompt,
adaptive KL (`target_kl=0.05`), sampling `T=1.0 / top_p=0.85 / rep_pen=1.05`.

Two scoring protocols appear below; they are **not** interchangeable.

- **clean-retry** — 50 test prompts, seed-0 first shot, then retry seeds
  100…111 until CER ≤ 0.30. Used for the reward-design ablation.
- **cross-axis** — 50 test prompts, single pass, every axis scored on every
  system. Used for the transfer matrix. This is the protocol behind
  `data/machine_scores/cross_axis_scores.csv` and the demo page's per-clip numbers.

---

## 1. Reward design: why the gate is a gate

Protocol: clean-retry. Δ is against the same base model on the same prompts.

| Setup | Reward | ΔAnimeScore | CER mean (median) | Violation rate | Improved |
|---|---|---|---|---|---|
| AS-only | raw AnimeScore, no CER | **+1.94** | 0.435 (0.113) | **34.0%** | — |
| scalar-CER | `0.6·AS + 0.4·max(0, 1−2·CER)` | +0.93 | 0.113 (0.046) | 24.0% | 40/50 |
| **zone-CER** | hard zones at 0.10 / 0.30, +0.5 / −1.0 | **+1.24** | **0.102 (0.041)** | **6.0%** | **43/50** |

Reading: AS-only follows the raw signal hardest and breaks a third of its
outputs. A soft CER penalty is a partial fix — violations fall to 24%, but half
the style gain goes with it. The hard gate compresses violations to 6% while
retaining 64% of the AS-only gain, and is the most consistent (43/50 improved).

> **Caveat, stated in the paper too:** the AS-only checkpoint was deleted before
> the clean-retry protocol was finalized, so its row is carried over from a
> prior 100-sentence non-clean evaluation. It is directionally right but not
> apples-to-apples with the other two rows.

---

## 2. Cross-axis transfer: rewards do not generalize across axes

Protocol: cross-axis. Rows = the axis the policy was trained on, columns = the
metric measured. Values are Δ against base.

| Trained on ↓ / Measured → | ΔCER | ΔAnimeScore | ΔLikability | ΔUTMOS |
|---|---|---|---|---|
| AnimeScore (step 1400) | −0.030 | **+1.353** | +0.075 | −0.037 |
| UTMOS (step 1020) | −0.030 | −0.072 | +0.140 | **+0.485** |
| Likability (step 540) | −0.018 | +0.029 | **+0.167** | +0.090 |

Base absolutes: CER 0.145 · AnimeScore −0.392 · Likability 4.205 · UTMOS 3.078.

The diagonal dominates in every row. Off-diagonal leakage is small, and the one
negative cell (AnimeScore training costs 0.037 UTMOS) is the expected tension
between stylized delivery and predicted naturalness. Optimizing one perceptual
axis does not quietly buy you the others.

---

## 3. Reward scale is why some axes barely move

| Axis | Backbone | Raw range | Base mean ± std | CV |
|---|---|---|---|---|
| AnimeScore | WavLM-base + RankNet | unbounded, ≈[−3, +5] | −0.39 ± 1.53 | **3.9** |
| UTMOS22-strong | SpeechMOS | MOS [1, 5] | 3.08 ± 1.01 | 0.33 |
| Likability | WavLM-base+ 6-class head | [1, 6] | 4.20 ± 0.44 | **0.10** |
| VAD arousal | wav2vec2-large-robust MSP-dim | ≈[0, 1] | 0.67 ± 0.13 | 0.19 |

Likability's base distribution is the narrowest by an order of magnitude
(CV 0.10), and it is also the axis that moves least under GRPO (+0.167). VAD
arousal is narrower still in the range that matters: its output span is small
enough that the −1.0 violation penalty dominates the gradient, and that run
(step 240, net +0.014 ≈ noise) is reported as a **negative result**. A reward
whose spread is small relative to its penalty cannot steer the policy.

---

## 4. Best-of-8 reranking vs GRPO

Protocol: cross-axis. BoN-8 reranks 8 base-model samples with the *same* zone
gate GRPO trains on, so the comparison isolates optimization (training vs
inference-time search) from reward specification.

| Axis | Δ base→GRPO | Δ base→BoN-8 |
|---|---|---|
| AnimeScore | **+1.353** | +1.078 |
| UTMOS | +0.485 | +0.461 |
| Likability | +0.167 | +0.145 |

GRPO beats an 8× inference budget on every axis, by the widest margin on the
axis with the widest base spread.

---

## 5. KL dynamics and checkpoint selection

Adaptive KL, `target_kl = 0.05`, `horizon = 2000`, in-reward path.

| | |
|---|---|
| β init → final | 0.050 → 0.082 |
| β plateau median (step ≥ 300) | 0.025 |
| Realized KL penalty, plateau median | 0.041 (18% under target) |
| Training steps | 1671 |

Validation over training:

| Step | val AS | val CER | val violation | val reward |
|---|---|---|---|---|
| 1050 | −0.138 | 0.230 | 0.27 | 1.902 |
| 1380 | +0.562 | 0.244 | 0.24 | 2.579 |
| **1400** | **+0.625** | 0.268 | 0.26 | **2.584** |
| 1710 | +0.955 | 0.308 | 0.31 | 2.556 |

Raw val AnimeScore keeps climbing past 1400 (0.63 → 0.96 by step 1710) — and so
does the violation rate (0.26 → 0.31). The constraint-aware reward peaks at
1400 and declines after. **Step 1400 is the selected checkpoint, and the gate is
what makes that visible**: without it the naive "AnimeScore is still going up"
signal would have picked 1710.

---

## Selected checkpoints

| Run | Axis | Step |
|---|---|---|
| `animescore_jp_zone_cer` | AnimeScore (JP) | 1400 |
| `animescore_en_zone_cer` | AnimeScore (EN) | 900 |
| `utmos_jp_zone_cer` | UTMOS | 1020 |
| `likability_jp_zone_cer` | Likability | 540 |
| `arousal_jp_zone_cer` | VAD arousal (failed) | 240 |
| `animescore_jp_scalar_cer` | AnimeScore, scalar-CER ablation | 480 |
| `animescore_jp_target_only` | AnimeScore, no CER ablation | 540 |

---

## Not in this release

Human listening-test results are part of the paper but not this repository: the
anonymized preference votes and their analysis code are held back. Every number
above is machine-side and recomputable from `data/machine_scores/` with the
scripts in `evaluation/paper_tables/`.
