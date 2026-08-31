"""Compute per-axis machine metrics (Δtarget, CER median, violation rate)
from a per-prompt scores CSV with columns idx,set,cer,animescore,likability,utmos.

Each `set` is one of {base, AS-GRPO, Likab-GRPO, UTMOS-GRPO} (or custom).
Δtarget is reported as the paired mean of (set - base) on the indicated
axis.

Outputs a small CSV (machine_baselines.csv schema): axis, method,
delta_target, cer_median, violation_rate, n_prompts, protocol.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from statistics import median, mean

TAU = 0.30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="cross_axis_scores.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--protocol", default="clean_single_seed", help="String for the 'protocol' column.")
    args = p.parse_args()

    by_set = defaultdict(list)
    with open(args.input) as f:
        for r in csv.DictReader(f):
            row = {"idx": int(r["idx"]),
                   "cer": float(r["cer"]) if r["cer"] not in ("","None") else None,
                   "animescore": float(r["animescore"]),
                   "likability": float(r["likability"]),
                   "utmos": float(r["utmos"])}
            by_set[r["set"]].append(row)

    base = {x["idx"]: x for x in by_set["base"]}
    out = []

    axis_map = {"AS-GRPO": "animescore", "UTMOS-GRPO": "utmos", "Likab-GRPO": "likability"}
    base_cer = [x["cer"] for x in by_set["base"] if x["cer"] is not None]
    out.append({"axis": "base", "method": "base",
                "delta_target": 0.0,
                "cer_median": median(base_cer),
                "violation_rate": sum(1 for c in base_cer if c > TAU)/len(base_cer),
                "n_prompts": len(base_cer), "protocol": args.protocol})

    for set_name, axis_key in axis_map.items():
        if set_name not in by_set: continue
        rows = by_set[set_name]
        diffs = []
        cers = []
        for x in rows:
            b = base.get(x["idx"])
            if b is None: continue
            diffs.append(x[axis_key] - b[axis_key])
            if x["cer"] is not None:
                cers.append(x["cer"])
        out.append({
            "axis": axis_key, "method": "zone_cer_grpo",
            "delta_target": mean(diffs),
            "cer_median": median(cers),
            "violation_rate": sum(1 for c in cers if c > TAU)/len(cers),
            "n_prompts": len(cers), "protocol": args.protocol,
        })

    with open(args.output, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out: w.writerow(r)
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
