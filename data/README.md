# Data

Everything the machine-side tables are computed from. No model weights, no audio
(the evaluation audio lives in `../docs/audio/`, where the demo page serves it).

```
prompts/          train / val / test prompt sets
  jp_train_prompts.csv    900   JP GRPO training
  jp_val_prompts.csv      100   JP validation
  jp_test_prompts.csv      50   JP test — the evaluation set
  en_train_prompts.csv    900   EN GRPO training
  en_val_prompts.csv       50 ⚠ identical to en_test_prompts.csv (see below)
  en_test_prompts.csv      50   EN test
machine_scores/
  cross_axis_scores.csv   200   every axis scored on every system (4 sets × 50)
  best_of_n_scores.csv    400   per-candidate scoring, 50 prompts × 8 seeds
  cer_retry_residuals.csv 200   what still failed after the 13-seed retry
manifests/
  audio_manifest.csv            sha256 + axis/side/index for every released clip
```

> ⚠ The EN run used its held-out test set as the validation set. This is a
> limitation acknowledged in the paper's appendix, not a packaging mistake — the
> files are released as they were used so the EN numbers can be read with that
> in mind.

## Schemas

**`prompts/*.csv`** — `idx, id, text`. `idx` (0-based) is the join key
everywhere else, and it is also the audio filename: `idx=7` ↔ `007.mp3`.

**`cross_axis_scores.csv`** — `set, idx, cer, animescore, likability, utmos`.
One row per (system, prompt). `set` ∈ `base`, `AS-GRPO`, `Likab-GRPO`,
`UTMOS-GRPO`. Every system is scored on every axis, which is what makes the
transfer matrix in `RESULTS.md` §2 possible: the off-diagonal cells are real
measurements, not estimates.

**`best_of_n_scores.csv`** — `idx, k, seed, cer, animescore, utmos, likability,
gated_reward`. Eight base-model candidates per prompt (`k` = 0…7).
`gated_reward` is the AnimeScore zone reward already applied, so
`argmax_k gated_reward` reproduces the BoN-8 selection for that axis directly;
for the other axes, `evaluation/paper_tables/compute_best_of_n.py` re-derives it
with the matching normalization.

**`cer_retry_residuals.csv`** — `axis, item_id, base_residual, grpo_residual,
union_residual, base_cer_after_retry, grpo_cer_after_retry`. `*_residual` is 1
when that side still exceeded CER 0.30 *after* all 13 retry seeds — i.e. the
failures that are not sampling luck.

## Not included

The human listening-test data (anonymized votes, per-item summaries, regression
inputs) and its analysis code are not part of this release. Every number in
`RESULTS.md` is machine-side and recomputable from the CSVs here.

## Use

Released for research inspection and replication. The prompt sets are
self-authored and free of third-party corpus licensing.
