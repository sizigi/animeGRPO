# Reward functions

verl `custom_reward_function` modules. Copy into `verl/utils/reward_score/` and
point a run at one:

```bash
custom_reward_function.path=verl/utils/reward_score/tts_animescore_cer_constraint.py
custom_reward_function.name=compute_score
```

Each returns a **dict**, not a scalar, so verl logs the components to wandb
separately — which is what makes "AnimeScore is still rising but so are
violations" visible during training rather than after it:

```python
{"score": float, "animescore": float, "cer": float, "cer_violation": float, "ok": float}
```

## The gate

`tts_animescore_cer_constraint.py` is the paper's main reward:

```python
as_pos = max(0.0, animescore + 3.0)      # base model's raw AS spans ≈[-3, +5]

if   cer >  0.30:  R = -1.0              # VIOLATE  — style gradient suppressed
elif cer <= 0.10:  R = as_pos + 0.5      # CLEAN
else:              R = as_pos            # FEASIBLE — optimize style here
```

The `+3.0` shift is not cosmetic. It keeps `violate < feasible ≤ clean` true for
every AnimeScore this base model actually produces. Without it, a rollout with a
strongly negative raw AnimeScore would score *below* the −1.0 violation penalty,
and the policy would learn that breaking the output is cheaper than speaking
badly — the exact failure the gate exists to prevent.

## Modules

| File | Axis | Gate |
|---|---|---|
| `tts_animescore_cer_constraint.py` | AnimeScore | zone (**main**) |
| `tts_utmos_cer_constraint.py` | UTMOS | zone |
| `tts_likability_cer_constraint.py` | Likability | zone |
| `tts_vad_arousal_cer_constraint.py` | VAD arousal | zone |
| `tts_animescore_cer.py` | AnimeScore | scalar weighted sum (ablation) |
| `tts_animescore.py` | AnimeScore | none (ablation) |
| `cer_zone.py` | — | reference implementation, no framework deps |

`cer_zone.py` is the same gate written without verl or HTTP, for reading and for
reuse in another trainer. The per-axis normalizations it expects:

| Axis | Raw → normalized |
|---|---|
| AnimeScore | `(x + 3.0) / 6.0` |
| UTMOS | `x / 5.0` |
| Likability | `(x − 1.0) / 5.0` |

Note the constraint modules apply the shift but not the `/6.0` scale — they feed
verl `max(0, AS + 3.0)` directly. The normalized form is what
`evaluation/paper_tables/compute_best_of_n.py` uses so that Best-of-N reranking
and GRPO share one reward specification across axes.

## Environment

```bash
export ANIMESCORE_SERVER=http://localhost:8002
export WHISPER_SERVER=http://localhost:8001
```

Run any module standalone to smoke-test it against live servers:

```bash
python tts_animescore_cer_constraint.py "<|s_130000|><|s_131000|>..." "こんにちは"
```
