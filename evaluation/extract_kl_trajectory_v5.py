"""Extract β (kl_coef) and KL trajectory from v5 zone-CER GRPO training logs.

Logs are 4 plain-text files covering global_step 1 → 1710 (with one restart at 510).
Each step line is `step:N - key:value - key:value - ...`.

Outputs:
  ~/samples/v5_kl_trajectory/
    trajectory.csv      — per-step β, kl_penalty, ppo_kl
    val_trajectory.csv  — per-eval-step val_AS, val_CER, val_violation, val_reward
    summary.json        — plateau β median + range + final values
    trajectory.png      — β + KL vs step (twin-axis)
    val_trajectory.png  — val AS/CER/violation vs step
"""
from pathlib import Path
import re, json, csv, statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("~/workspace/llasa_rl/logs").expanduser()
LOGS = [
    LOG_DIR / "train_v5_constrained.log",  # step 0-530
    LOG_DIR / "train_v5_resume.log",       # step 510-840
    LOG_DIR / "train_v5_extend.log",       # step 841-1400
    LOG_DIR / "train_v5_extend2.log",      # step 1400-1730
]
OUT = Path("~/samples/v5_kl_trajectory").expanduser()
OUT.mkdir(parents=True, exist_ok=True)

# Keys we care about
TRAIN_KEYS = {
    "training/global_step":          "step",
    "actor/reward_kl_penalty_coeff": "beta",
    "actor/reward_kl_penalty":       "kl_penalty",  # β·KL (or KL term applied to reward)
    "actor/ppo_kl":                  "ppo_kl",
    "actor/entropy_loss":            "entropy",
    "critic/score/mean":             "train_reward",
}
VAL_KEYS = {
    "val-core/jp_tts_mixed_v2/reward/mean@1":         "val_reward",
    "val-aux/jp_tts_mixed_v2/animescore/mean@1":      "val_AS",
    "val-aux/jp_tts_mixed_v2/cer/mean@1":             "val_CER",
    "val-aux/jp_tts_mixed_v2/cer_violation/mean@1":   "val_violation",
}

# Match "key:numeric" preserving negative + decimal
PAIR = re.compile(r"([A-Za-z0-9_\-/@]+):(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
STEP_PREFIX = re.compile(r"step:(\d+)")


def parse_logs():
    train_rows, val_rows = [], []
    for path in LOGS:
        if not path.exists():
            print(f"  [skip] {path} (missing)")
            continue
        n_train = n_val = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "step:" not in line:
                    continue
                # Strip the "(TaskRunner pid=...) " prefix if present
                if ") step:" in line:
                    line = line.split(") ", 1)[1]
                pairs = dict(PAIR.findall(line))
                if not pairs:
                    continue
                # Validation rows: have val-core key, lack training/global_step
                if any(k in pairs for k in VAL_KEYS):
                    sm = STEP_PREFIX.search(line)
                    step = int(sm.group(1)) if sm else None
                    row = {"step": step, "_src": path.name}
                    for k, alias in VAL_KEYS.items():
                        row[alias] = float(pairs[k]) if k in pairs else None
                    val_rows.append(row); n_val += 1
                # Training rows: have training/global_step
                elif "training/global_step" in pairs:
                    row = {"_src": path.name}
                    for k, alias in TRAIN_KEYS.items():
                        row[alias] = float(pairs[k]) if k in pairs else None
                    row["step"] = int(row.pop("step")) if row.get("step") is not None else None
                    train_rows.append(row); n_train += 1
        print(f"  {path.name}: {n_train} train rows, {n_val} val rows")
    return train_rows, val_rows


def dedupe_by_step(rows):
    """Resumes overlap (e.g. constrained ends 530, resume starts 510). Keep the
    LATER log's row for any step seen twice — that's the continuing run."""
    by_step = {}
    src_order = {p.name: i for i, p in enumerate(LOGS)}
    for r in rows:
        if r["step"] is None:
            continue
        prev = by_step.get(r["step"])
        if prev is None or src_order[r["_src"]] >= src_order[prev["_src"]]:
            by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def write_csv(rows, path, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_kl(train_rows, val_rows):
    steps   = [r["step"]       for r in train_rows]
    beta    = [r.get("beta")   for r in train_rows]
    klpen   = [r.get("kl_penalty") for r in train_rows]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(steps, beta, color="C0", lw=1.2, label="β  (kl_coef)")
    ax1.set_xlabel("global step")
    ax1.set_ylabel("β  (kl_coef)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.plot(steps, klpen, color="C3", lw=0.8, alpha=0.7, label="reward_kl_penalty")
    ax2.set_ylabel("reward_kl_penalty (β·KL)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")

    # Mark adaptive-KL target
    ax1.axhline(0.05, color="C0", ls="--", lw=0.6, alpha=0.5)
    # Mark the protected best ckpt (step 1400)
    ax1.axvline(1400, color="k", ls=":", lw=0.6, alpha=0.5)
    ax1.text(1400, ax1.get_ylim()[1]*0.95, " step 1400 (best ckpt)", fontsize=8, va="top")
    fig.suptitle("v5 zone-CER GRPO — adaptive-KL β and KL trajectory")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.png", dpi=150)
    plt.close(fig)


def plot_val(val_rows):
    val_rows = [r for r in val_rows if r["step"] is not None]
    steps = [r["step"] for r in val_rows]
    AS    = [r.get("val_AS")        for r in val_rows]
    CER   = [r.get("val_CER")       for r in val_rows]
    VIO   = [r.get("val_violation") for r in val_rows]
    REW   = [r.get("val_reward")    for r in val_rows]

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax[0,0].plot(steps, AS,  color="C2"); ax[0,0].set_title("val AnimeScore (mean)")
    ax[0,1].plot(steps, REW, color="C0"); ax[0,1].set_title("val reward (mean)")
    ax[1,0].plot(steps, CER, color="C3"); ax[1,0].set_title("val CER (mean)")
    ax[1,0].axhline(0.30, color="k", ls="--", lw=0.5)
    ax[1,1].plot(steps, VIO, color="C1"); ax[1,1].set_title("val CER violation rate")
    for a in ax.flat: a.grid(alpha=0.3); a.set_xlabel("global step")
    fig.suptitle("v5 zone-CER GRPO — validation trajectory")
    fig.tight_layout()
    fig.savefig(OUT / "val_trajectory.png", dpi=150)
    plt.close(fig)


def main():
    print("== Parsing logs ==")
    tr_raw, vl_raw = parse_logs()
    tr = dedupe_by_step(tr_raw)
    vl = dedupe_by_step(vl_raw)
    print(f"\nDeduped: {len(tr)} train steps, {len(vl)} val steps")
    print(f"Train step range: {tr[0]['step']} → {tr[-1]['step']}")
    if vl:
        print(f"Val   step range: {vl[0]['step']} → {vl[-1]['step']}")

    write_csv(tr, OUT/"trajectory.csv",
              ["step","beta","kl_penalty","ppo_kl","entropy","train_reward","_src"])
    write_csv(vl, OUT/"val_trajectory.csv",
              ["step","val_reward","val_AS","val_CER","val_violation","_src"])

    # β plateau analysis
    betas = [r["beta"] for r in tr if r.get("beta") is not None]
    # Plateau = late phase, post-warmup. Use step >= 300 (post-initial-adaptation)
    plateau = [(r["step"], r["beta"]) for r in tr
               if r.get("beta") is not None and r["step"] >= 300]
    plateau_only = [b for _, b in plateau]
    early = [r["beta"] for r in tr if r.get("beta") is not None and r["step"] < 300]

    summary = {
        "n_train_steps": len(tr),
        "n_val_steps":   len(vl),
        "step_range":    [tr[0]["step"], tr[-1]["step"]],
        "target_kl":     0.05,
        "adaptive_horizon": 2000,
        "beta": {
            "init":            betas[0],
            "final":           betas[-1],
            "min":             round(min(betas), 6),
            "max":             round(max(betas), 6),
            "overall_median":  round(st.median(betas), 6),
            "overall_mean":    round(st.mean(betas), 6),
            "early_median (step<300)":   round(st.median(early), 6) if early else None,
            "plateau_median (step>=300)": round(st.median(plateau_only), 6),
            "plateau_mean":              round(st.mean(plateau_only), 6),
            "plateau_p25":               round(st.quantiles(plateau_only, n=4)[0], 6),
            "plateau_p75":               round(st.quantiles(plateau_only, n=4)[2], 6),
        },
    }
    # Add KL penalty stats
    klpens = [r["kl_penalty"] for r in tr if r.get("kl_penalty") is not None]
    if klpens:
        plat_kl = [r["kl_penalty"] for r in tr
                   if r.get("kl_penalty") is not None and r["step"] >= 300]
        summary["reward_kl_penalty"] = {
            "final":           round(klpens[-1], 6),
            "plateau_median":  round(st.median(plat_kl), 6),
            "plateau_mean":    round(st.mean(plat_kl), 6),
            "max":             round(max(klpens), 6),
        }
    # Val metrics at the final ckpt step (use step 1400, the best one)
    val_at = {r["step"]: r for r in vl}
    for s in (1050, 1380, 1400, 1710):
        if s in val_at:
            summary[f"val@step_{s}"] = {
                k: val_at[s].get(k) for k in ("val_AS","val_CER","val_violation","val_reward")
            }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n== Summary ==")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    plot_kl(tr, vl)
    plot_val(vl)
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
