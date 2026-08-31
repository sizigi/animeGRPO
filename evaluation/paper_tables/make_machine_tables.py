"""Drive the three machine-table outputs in one command.

Reads data/machine_scores/cross_axis_scores.csv (per-prompt scoring) and
data/machine_scores/best_of_n_scores.csv (per-candidate scoring) and emits:
  - machine_baselines.csv  (Table 2 / tab:ablation3way + BoN scaling rows)
  - cross_axis_table.csv   (Table 3 / tab:heatmap4 cells)
  - bon_scaling.csv        (Best-of-N N in {1,2,4,8} scaling per axis)

This is a thin wrapper that calls compute_machine_metrics.py,
compute_cross_axis_table.py, and compute_best_of_n.py with consistent
defaults.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to data/ (containing machine_scores/)")
    p.add_argument("--out_dir",  required=True)
    args = p.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent

    cross_in = Path(args.data_dir) / "machine_scores" / "cross_axis_scores.csv"
    bon_in   = Path(args.data_dir) / "machine_scores" / "best_of_n_scores.csv"

    cmd1 = [sys.executable, str(here/"compute_machine_metrics.py"),
            "--input", str(cross_in), "--output", str(out/"machine_baselines.csv")]
    cmd2 = [sys.executable, str(here/"compute_cross_axis_table.py"),
            "--input", str(cross_in), "--output", str(out/"cross_axis_table.csv")]
    for c in (cmd1, cmd2):
        print(" ".join(c))
        subprocess.run(c, check=True)

    # BoN scaling per axis
    for axis in ("animescore", "utmos", "likability"):
        cmd = [sys.executable, str(here/"compute_best_of_n.py"),
               "--input", str(bon_in),
               "--output", str(out/f"bon_scaling_{axis}.csv"),
               "--axis", axis, "--N", "8"]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nDone. Outputs in {out}")


if __name__ == "__main__":
    main()
