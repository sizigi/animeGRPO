"""Machine-side evaluation scripts.

- compute_machine_metrics.py: per-axis Δtarget / CER median / violation rate.
- compute_cer_retry.py: 13-seed symmetric CER-retry protocol (canonical).
- compute_best_of_n.py: BoN reranking using the zone-CER reward gate.
- compute_cross_axis_table.py: 3x4 cross-axis heatmap.
- make_machine_tables.py: orchestrate the three above into the paper's
  machine tables.
"""
