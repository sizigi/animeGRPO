"""Produce the cross-axis 3x4 heatmap (Table 'heatmap4' in the paper).

Each row is a GRPO-trained policy (Anime / Likab / UTMOS) and each column
is a metric (CER, AnimeScore, Likability, UTMOS). The cell value is the
paired mean of (GRPO row - base) on the column metric.

Inputs:  machine_scores/cross_axis_scores.csv (long-format, columns:
         set, idx, cer, animescore, likability, utmos).
Output:  CSV with rows {reward, delta_cer, delta_animescore, delta_likability, delta_utmos}.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from statistics import mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    by_set = defaultdict(list)
    with open(args.input) as f:
        for r in csv.DictReader(f):
            by_set[r["set"]].append({"idx": int(r["idx"]),
                                     "cer": float(r["cer"]) if r["cer"] not in ("","None") else None,
                                     "animescore": float(r["animescore"]),
                                     "likability": float(r["likability"]),
                                     "utmos": float(r["utmos"])})

    base = {x["idx"]: x for x in by_set["base"]}
    row_map = {"AS-GRPO":"Anime", "Likab-GRPO":"Likab", "UTMOS-GRPO":"UTMOS"}

    rows = []
    for set_name, label in row_map.items():
        if set_name not in by_set: continue
        d_cer, d_as, d_lik, d_utm = [], [], [], []
        for x in by_set[set_name]:
            b = base.get(x["idx"])
            if b is None: continue
            if x["cer"] is not None and b["cer"] is not None:
                d_cer.append(x["cer"] - b["cer"])
            d_as.append(x["animescore"] - b["animescore"])
            d_lik.append(x["likability"] - b["likability"])
            d_utm.append(x["utmos"] - b["utmos"])
        rows.append({"reward": label,
                     "delta_cer": mean(d_cer), "delta_animescore": mean(d_as),
                     "delta_likability": mean(d_lik), "delta_utmos": mean(d_utm)})

    with open(args.output, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
