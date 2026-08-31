"""Best-of-N reranking using the same CER-zone gated reward as GRPO training.

For each prompt:
  1. Generate N candidates from the base model with distinct seeds.
  2. Score each candidate's CER and axis reward; compute the gated reward
     R = norm(axis) + 0.5  if CER <= 0.10
     R = norm(axis)         if 0.10 < CER <= 0.30
     R = -1.0               otherwise.
  3. Return argmax_N R, keeping CER metadata for downstream analysis.

This is the BoN baseline reported in the paper. The reward gate is identical
to what GRPO uses during training, so BoN-N and GRPO use the same reward
specification — only the optimization (training vs inference reranking)
differs.

Input CSV columns (per-candidate scoring): idx,k,seed,cer,animescore,utmos,
likability,gated_reward (the table machine_scores/best_of_n_scores.csv in
the data release has this schema).
Output CSV columns: idx,axis,N,chosen_k,chosen_seed,chosen_cer,chosen_axis_score.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict


TAU_HIGH, TAU_LOW = 0.30, 0.10
RHO, BONUS = 1.0, 0.5


def gated(score_norm: float, cer: float | None) -> float:
    if cer is None or cer > TAU_HIGH:
        return -RHO
    base = max(0.0, float(score_norm))
    return base + BONUS if cer <= TAU_LOW else base


def normalize(axis: str, score: float) -> float:
    if axis == "animescore":
        return (score + 3.0) / 6.0
    if axis == "utmos":
        return score / 5.0
    if axis == "likability":
        return (score - 1.0) / 5.0
    raise ValueError(axis)


def select_bon(candidates, axis: str, N: int):
    """candidates is a list of dicts (already sorted by k=0..7).
    Returns the dict whose gated reward is highest among the first N.
    """
    pool = candidates[:N]
    return max(pool, key=lambda c: gated(normalize(axis, c[axis]), c["cer"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="best_of_n_scores.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--axis", choices=("animescore","utmos","likability"), required=True)
    p.add_argument("--N", type=int, default=8)
    args = p.parse_args()

    by_idx = defaultdict(list)
    with open(args.input) as f:
        for row in csv.DictReader(f):
            row["k"]   = int(row["k"])
            row["cer"] = float(row["cer"]) if row["cer"] not in ("", None) else None
            for k in ("animescore","utmos","likability","gated_reward"):
                if row[k] not in ("", None):
                    row[k] = float(row[k])
            by_idx[int(row["idx"])].append(row)

    out_rows = []
    for idx, cands in sorted(by_idx.items()):
        cands.sort(key=lambda c: c["k"])
        for N in [1, 2, 4, 8]:
            if N > args.N: continue
            best = select_bon(cands, args.axis, N)
            out_rows.append({
                "idx": idx, "axis": args.axis, "N": N,
                "chosen_k": best["k"], "chosen_seed": best.get("seed"),
                "chosen_cer": best["cer"], "chosen_axis_score": best[args.axis],
            })

    with open(args.output, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {len(out_rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
