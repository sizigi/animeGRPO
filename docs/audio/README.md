# Evaluation audio

The clips behind the [demo page](https://sizigi.github.io/animeGRPO/), and the
exact audio that was served to listeners in the paper's pairwise evaluation.

```
jp_animescore/{base, grpo, best_of_8}     50 each   AnimeScore zone-CER, step 1400
jp_utmos/{base, grpo, best_of_8}          50 each   UTMOS zone-CER, step 1020
jp_likability/{base, grpo, best_of_8}     50 each   Likability zone-CER, step 540
en_animescore/{base, grpo}                50 each   AnimeScore zone-CER (EN), step 900
```

550 files. Mono, 16 kHz, MP3 at 48 kbps — transcoded from the served PCM wavs to
keep the repository small. `NNN.mp3` is prompt `idx = NNN` in
`../../data/prompts/jp_test_prompts.csv` (or `en_test_prompts.csv`), and the
per-clip scores are in `../../data/machine_scores/cross_axis_scores.csv`.

- **base** — `HKUSTAudio/Llasa-1B-Multilingual`, seed 0, after the 13-seed CER retry.
- **grpo** — the selected constrained checkpoint for that axis.
- **best_of_8** — the best of eight base-model samples under the same CER-zone
  gate GRPO trains on. Only the chosen candidate ships; the full 8-candidate
  pool is summarized in `best_of_n_scores.csv`.

SHA-256 for every file: `../../data/manifests/audio_manifest.csv` (paths there
are relative to the original release layout, `data/audio_samples/…`, and name
the pre-transcode `.wav` sources).

## Responsible use

Released for research inspection of the paper's results. **Must not be used for
impersonation or unauthorized style imitation** of any voice, real or fictional.
Some clips were synthesized under reward objectives that push delivery toward
anime voice-acting performance; do not redeploy them outside this
research-replication context.
