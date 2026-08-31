"""Apples-to-apples eval of UTMOS-only (step 420) on the same test_v2 + clean
protocol as v5 zone-CER and v6 AS-only. Output mirrors the v5 / v6 summary
schemas.

Two-pass:
  Pass 1 (raw): seed=0 only on all 50 prompts -> CER violation rate, raw AS/UTMOS.
  Pass 2 (clean): For samples with seed-0 CER > 0.30, retry seeds 100..111 until
                  CER <= 0.30 or seeds exhausted. Keep lowest-CER seed.
  Scoring: AS, UTMOS via /score {"tokens": ...}; CER via Whisper /score.

Base column is reused from clean_likability_step540_summary.json
(per_sample.base has seed/cer/animescore/utmos/likability for the SAME 50 prompts
on the SAME HKUSTAudio/Llasa-1B-Multilingual base).
"""
import os, json, time, gc, warnings
from pathlib import Path
import torch, numpy as np, soundfile as sf, pandas as pd, requests
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from xcodec2.modeling_xcodec2 import XCodec2Model
from jiwer import cer as compute_cer

warnings.filterwarnings("ignore")

ROOT = Path("~/samples/llasa_1b_utmos_only_step420_eval").expanduser()
ROOT.mkdir(parents=True, exist_ok=True)
TEST_PARQUET = os.path.expanduser("~/data/llasa-tts-rl-grpo-jp-v2/test_v2.parquet")
BASE_SUMMARY = os.path.expanduser("~/samples/llasa_1b_likability_step540_eval/clean_likability_step540_summary.json")

CKPT = os.path.expanduser("~/checkpoints/llasa-1b-utmos-only-step420-merged")
TAG  = "utmos_only_420"
REWARD_DESC = "tts_utmos.py (raw UTMOS/5.0, no CER gate) — UTMOS-axis CER-zone ablation"

NUM_EVAL = 50
CER_THRESHOLD = 0.30
RETRY_SEEDS = list(range(100, 112))
DEVICE = "cuda:0"

WHISPER    = os.getenv("WHISPER_SERVER",    "http://localhost:8001")
ANIMESCORE = os.getenv("ANIMESCORE_SERVER", "http://localhost:8002")
UTMOS_SRV  = os.getenv("UTMOS_SERVER",      "http://localhost:8007")

SPEECH_TOKEN_START = 128264
SPEECH_TOKEN_END   = 193799
SPEECH_END_ID      = 128261


class SpeechOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids, vocab_size):
        self.mask = torch.full((vocab_size,), float("-inf"))
        for i in allowed_ids:
            self.mask[i] = 0.0
    def __call__(self, input_ids, scores):
        return scores + self.mask.to(scores.device)


def remote(url, payload, timeout=180):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [server err] {url}: {e}")
        return None


def score_cer(tokens, gt_text):
    r = remote(f"{WHISPER}/score", {"tokens": tokens, "text": gt_text})
    if not r: return 1.0, ""
    transcript = r.get("transcript", "")
    c = float(compute_cer(gt_text, transcript)) if transcript else 1.0
    return c, transcript


def score_as(tokens):
    r = remote(f"{ANIMESCORE}/score", {"tokens": tokens})
    return float(r["raw_score"]) if r and "raw_score" in r else None


def score_utmos(tokens):
    r = remote(f"{UTMOS_SRV}/score", {"tokens": tokens})
    if not r: return None
    return float(r.get("utmos") or r.get("raw_score") or r.get("score"))


def load_prompts():
    df = pd.read_parquet(TEST_PARQUET).head(NUM_EVAL).reset_index(drop=True)
    return [{"idx": int(i), "id": r["extra_info"]["id"], "text": r["extra_info"]["text"]}
            for i, r in df.iterrows()]


def load_base_per_sample():
    j = json.load(open(BASE_SUMMARY))
    return {p["idx"]: p["base"] for p in j["per_sample"]}


def gen(model, tokenizer, codec, text, logits_proc, seed,
        max_new=2048, top_p=0.85, temperature=1.0, repetition_penalty=1.05):
    chat = [
        {"role":"user","content":f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"},
        {"role":"assistant","content":"<|SPEECH_GENERATION_START|>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", continue_final_message=True,
    ).to(DEVICE)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=SPEECH_END_ID,
            pad_token_id=tokenizer.pad_token_id or SPEECH_END_ID,
            logits_processor=[logits_proc],
        )
    new = out[0, input_ids.shape[1]:].tolist()
    if new and new[-1] == SPEECH_END_ID:
        new = new[:-1]
    ids = [t - SPEECH_TOKEN_START for t in new
           if SPEECH_TOKEN_START <= t <= SPEECH_TOKEN_END]
    if not ids: return None, []
    code_t = torch.tensor([[ids]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        wav = codec.decode_code(code_t).cpu().numpy().squeeze()
    return wav, ids


def main():
    print(f"Loading prompts from {TEST_PARQUET}")
    rows = load_prompts()
    base_ps = load_base_per_sample()
    print(f"  {len(rows)} prompts; base reused from {Path(BASE_SUMMARY).name}")

    print(f"Loading ckpt {CKPT}")
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    allowed = list(range(SPEECH_TOKEN_START, SPEECH_TOKEN_END + 1)) + [SPEECH_END_ID]
    logits_proc = SpeechOnlyLogitsProcessor(allowed, model.config.vocab_size)
    print("Loading XCodec2")
    codec = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2").to(DEVICE).eval()

    print("\n=== PASS 1: raw seed=0 (for violation rate) ===")
    raw_dir = ROOT / "raw_seed0"; raw_dir.mkdir(exist_ok=True)
    raw = {}
    for r in rows:
        t0 = time.time()
        wav, ids = gen(model, tokenizer, codec, r["text"], logits_proc, seed=0)
        if not ids:
            print(f"  [{r['idx']:02d}] EMPTY"); raw[r["idx"]] = None; continue
        c, transcript = score_cer(ids, r["text"])
        a  = score_as(ids)
        u  = score_utmos(ids)
        sf.write(str(raw_dir / f"{r['idx']:03d}.wav"), wav, 16000)
        raw[r["idx"]] = {"seed":0, "cer":c, "animescore":a, "utmos":u, "tokens":ids,
                         "transcript":transcript}
        a_s = f"{a:+.2f}" if a is not None else "NA"
        u_s = f"{u:.2f}"  if u is not None else "NA"
        print(f"  [{r['idx']:02d}] seed=0  CER={c:.3f}  AS={a_s}  UTMOS={u_s}  ({time.time()-t0:.1f}s)")

    raw_valid = [v for v in raw.values() if v]
    raw_cers  = [v["cer"] for v in raw_valid]
    n_viol    = sum(1 for c in raw_cers if c > CER_THRESHOLD)
    print(f"\n  raw seed=0: {len(raw_valid)}/{NUM_EVAL} valid, CER mean={np.mean(raw_cers):.3f} median={np.median(raw_cers):.3f}, violation>{CER_THRESHOLD}: {n_viol}/{len(raw_valid)} ({100*n_viol/max(1,len(raw_valid)):.1f}%)")

    print(f"\n=== PASS 2: clean (retry seeds for samples with seed-0 CER > {CER_THRESHOLD}) ===")
    clean_dir = ROOT / f"clean_{TAG}"; clean_dir.mkdir(exist_ok=True)
    chosen = {}
    for r in rows:
        i = r["idx"]
        seed0 = raw[i]
        if seed0 and seed0["cer"] <= CER_THRESHOLD:
            sf.write(str(clean_dir / f"{i:03d}.wav"),
                     sf.read(str(raw_dir / f"{i:03d}.wav"))[0], 16000)
            chosen[i] = {**seed0, "from": "seed0"}
            continue
        best = seed0
        seed0_cer_str = f"{seed0['cer']:.3f}" if seed0 else "NA"
        print(f"  [{i:02d}] retry (seed0 CER={seed0_cer_str})")
        for s in RETRY_SEEDS:
            t0 = time.time()
            wav, ids = gen(model, tokenizer, codec, r["text"], logits_proc, seed=s)
            if not ids: continue
            c, transcript = score_cer(ids, r["text"])
            print(f"      seed={s:3d} CER={c:.3f} ({time.time()-t0:.1f}s)")
            if best is None or c < best["cer"]:
                a = score_as(ids)
                u = score_utmos(ids)
                best = {"seed":s, "cer":c, "animescore":a, "utmos":u,
                        "tokens":ids, "transcript":transcript, "wav":wav}
            if c <= CER_THRESHOLD:
                break
        if best is None:
            print(f"  [{i:02d}] FAILED ALL"); continue
        if "wav" in best:
            sf.write(str(clean_dir / f"{i:03d}.wav"), best["wav"], 16000)
            best.pop("wav")
        chosen[i] = {**best, "from": "retry"}

    del model, codec
    gc.collect(); torch.cuda.empty_cache()

    print("\n=== Aggregate ===")
    per_sample = []
    for r in rows:
        i = r["idx"]
        b = base_ps.get(i, {})
        v = chosen.get(i, {})
        rr = raw.get(i, {}) or {}
        per_sample.append({
            "idx": i, "text_id": r["id"], "text": r["text"],
            "base": {"seed": b.get("seed"), "cer": b.get("cer"),
                     "likability": b.get("likability"),
                     "animescore": b.get("animescore"),
                     "utmos": b.get("utmos")},
            f"{TAG}_raw_seed0": {"cer": rr.get("cer"),
                                 "animescore": rr.get("animescore"),
                                 "utmos": rr.get("utmos")},
            f"{TAG}_clean":    {"seed": v.get("seed"), "cer": v.get("cer"),
                                "animescore": v.get("animescore"),
                                "utmos": v.get("utmos"),
                                "from": v.get("from")},
        })

    def stat(xs, key=None):
        if key is not None:
            xs = [x[key] for x in xs if x.get(key) is not None]
        xs = [x for x in xs if x is not None]
        if not xs: return None
        return {"n": len(xs),
                "mean": float(np.mean(xs)), "std": float(np.std(xs)),
                "median": float(np.median(xs)),
                "min": float(np.min(xs)), "max": float(np.max(xs))}

    base_list = [p["base"] for p in per_sample]
    raw_list  = [p[f"{TAG}_raw_seed0"] for p in per_sample]
    cln_list  = [p[f"{TAG}_clean"]     for p in per_sample]

    summary = {
        "n": NUM_EVAL,
        "ckpt": CKPT,
        "reward": REWARD_DESC,
        "cer_threshold": CER_THRESHOLD,
        "base_summary":     {k: stat(base_list, k) for k in ("cer","animescore","utmos","likability")},
        f"{TAG}_raw_seed0": {k: stat(raw_list,  k) for k in ("cer","animescore","utmos")},
        f"{TAG}_clean":     {k: stat(cln_list,  k) for k in ("cer","animescore","utmos")},
        "violation_rate_seed0": {
            "n_violations": int(sum(1 for x in raw_list
                                    if x.get("cer") is not None and x["cer"] > CER_THRESHOLD)),
            "n_valid":      int(sum(1 for x in raw_list if x.get("cer") is not None)),
        },
        "per_sample": per_sample,
    }
    summary["violation_rate_seed0"]["rate"] = (
        summary["violation_rate_seed0"]["n_violations"]
        / max(1, summary["violation_rate_seed0"]["n_valid"])
    )

    deltas_AS    = [p[f"{TAG}_clean"].get("animescore",0) - p["base"].get("animescore",0)
                    for p in per_sample
                    if p[f"{TAG}_clean"].get("animescore") is not None and p["base"].get("animescore") is not None]
    deltas_UTMOS = [p[f"{TAG}_clean"].get("utmos",0) - p["base"].get("utmos",0)
                    for p in per_sample
                    if p[f"{TAG}_clean"].get("utmos") is not None and p["base"].get("utmos") is not None]
    summary["delta_clean"] = {
        "animescore": stat(deltas_AS) if deltas_AS else None,
        "utmos":      stat(deltas_UTMOS) if deltas_UTMOS else None,
        "n_improved_AS": int(sum(1 for d in deltas_AS if d > 0)),
        "n_improved_UTMOS": int(sum(1 for d in deltas_UTMOS if d > 0)),
    }

    out = ROOT / "summary.json"
    json.dump(summary, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")
    print(json.dumps({k: summary[k] for k in
                      ("base_summary", f"{TAG}_raw_seed0", f"{TAG}_clean",
                       "violation_rate_seed0", "delta_clean")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
