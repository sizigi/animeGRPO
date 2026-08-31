# Evaluation

Two independent layers:

- **`paper_tables/`** recomputes paper tables from the CSVs in `../data/` with
  nothing but Python and a CSV reader. No GPU, no model, no reward server.
- **The scripts in this directory** regenerate audio from a checkpoint and
  re-score it. These need the policy, XCodec2, and the reward servers running.

If you want to check the paper's arithmetic, use the first. If you want to
reproduce a checkpoint's numbers, use the second.

---

## Recomputing paper tables (no GPU)

```bash
cd evaluation/paper_tables

python compute_cross_axis_table.py \
    --input  ../../data/machine_scores/cross_axis_scores.csv \
    --output heatmap.csv                      # → RESULTS.md §2

python compute_machine_metrics.py \
    --input  ../../data/machine_scores/cross_axis_scores.csv \
    --output baselines.csv

for axis in animescore utmos likability; do
  python compute_best_of_n.py \
      --input ../../data/machine_scores/best_of_n_scores.csv \
      --axis "$axis" --N 8 --output bon_$axis.csv   # → RESULTS.md §4
done

python compute_cer_retry.py \
    --input  ../../data/machine_scores/cer_retry_residuals.csv \
    --output retry.csv
```

`compute_best_of_n.py` is worth reading even if you never run it: it applies the
*same* zone gate GRPO trains on, so BoN-N and GRPO differ only in optimization
(inference-time search vs training), not in what they are optimizing.

Numbers match `../RESULTS.md` to three decimals, because every table there
derives from these same CSVs.

---

## Regenerating from a checkpoint (GPU + reward servers)

**1. Merge** the FSDP checkpoint into a HuggingFace model:

```bash
python merge_1b_grpo.py \
    --ckpt_dir   checkpoints/<exp>/global_step_1400 \
    --output_dir ~/checkpoints/llasa-1b-v5-step1400-merged
```

Single-GPU FSDP stores the actor as one shard with DTensor-wrapped values;
the script unwraps `_local_tensor` and loads it into the base architecture
`strict=True`, so a silent key mismatch fails loudly rather than producing a
half-initialized model.

**2. Start the reward servers** (Whisper :8001, AnimeScore :8002, and whichever
axis you are scoring) — see `../training/reward_servers/README.md`.

**3. Evaluate:**

| Script | Run |
|---|---|
| `evaluate_1b_v5_test_v2.py` | zone-CER, JP (step 1400) |
| `evaluate_v4_test_v2_clean.py` | scalar-CER ablation (step 480) |
| `evaluate_v6_test_v2_clean.py` | AS-only ablation (step 540) |
| `evaluate_utmos_only_test_v2_clean.py` | UTMOS target-only (step 420) |
| `evaluate_bon_llasa1b.py` | Best-of-N baseline over base-model samples |

### The clean-retry protocol

The ablation scripts run two passes, and the second is what makes the
comparison fair:

1. **raw** — seed 0 on all 50 prompts. Gives the violation rate: the fraction
   with CER > 0.30.
2. **clean** — for every prompt that violated, retry seeds 100…111 until CER
   ≤ 0.30, keeping the lowest-CER result.

Without the retry pass, a system that fails loudly on 30% of prompts and a
system that fails on 6% are compared on incomparable sets of audio: the mean
style score of the first is dominated by whether its garbage happens to score
well. The retry pass puts both systems on outputs a listener could actually
sit through, and the violation rate is reported separately as its own number.

### Analysis helpers

| Script | Output |
|---|---|
| `extract_kl_trajectory_v5.py` | β and KL-penalty trajectory from training logs → `RESULTS.md` §5 |
| `plot_3way_ablation.py` | the reward-design ablation figure |

---

## A note on paths

These scripts are the ones that produced the paper's numbers, kept as they ran
rather than rewritten into a general-purpose CLI. Inputs and outputs default to
`~/data/...`, `~/checkpoints/...`, and `~/samples/...`; adjust the constants at
the top of each file. Prompt sets are also exported as CSV in `../data/prompts/`
if you would rather not rebuild the parquet.
