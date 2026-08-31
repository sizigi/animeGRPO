"""Symmetric 13-seed CER-retry protocol used for human-evaluation audio.

For each prompt and each side (base / GRPO), we try seed=0 first. If the
generated sample has CER > tau_high (=0.30), we try seeds 100..111 (12
additional seeds) in order, keeping the first sample whose CER <= tau_high.
If all 13 candidates fail, we keep the lowest-CER candidate. The same rule
applies symmetrically to base and GRPO sides; all 50 items are retained for
human evaluation, including any residual violators.

Input:   per-prompt seed-0 generations + on-demand retry seeds, with a CER
         scorer (Whisper /score) and the axis reward scorer.
Output:  one (idx, side, seed, cer, axis_score) tuple per item per side.

This script is a CLI wrapper around the retry loop; it expects the caller
to provide a generation function and a CER scoring function. We do not
include audio synthesis here (see reward_functions/ for which servers we
hit and docs/reproducibility_notes.md for the policy checkpoint manifest).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

TAU_HIGH = 0.30
RETRY_SEEDS = list(range(100, 112))


def run_retry(idx: int, side: str, generate_fn, score_cer_fn,
              tau_high: float = TAU_HIGH, seeds=(0, *RETRY_SEEDS)):
    """Symmetric retry. Returns dict {seed, cer, ...}.

    Args:
        generate_fn: callable(seed) -> dict with at least key 'cer' (must be
            scored externally by score_cer_fn) and any axis-reward fields.
    """
    best = None
    for s in seeds:
        cand = generate_fn(s)
        cand["seed"] = s
        if "cer" not in cand:
            cand["cer"] = score_cer_fn(cand)
        if best is None or (cand["cer"] is not None and (best["cer"] is None or cand["cer"] < best["cer"])):
            best = cand
        if cand.get("cer") is not None and cand["cer"] <= tau_high:
            return best
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="JSON with keys: prompts (list of {idx,text}), sides "
                        "(list of model paths), output_dir.")
    args = p.parse_args()
    cfg = json.load(open(args.config))
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    print("Loaded config; supply your own generate_fn / score_cer_fn at the "
          "callsite (e.g., a Llasa-1B-Multilingual `generate(seed)` wrapper and the "
          "openai/whisper-large-v3 CER server at the URL in your env). The retry "
          "loop above is the canonical protocol.")


if __name__ == "__main__":
    main()
