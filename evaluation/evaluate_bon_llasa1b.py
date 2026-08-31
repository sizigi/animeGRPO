"""Best-of-N baseline on Llasa-1B-Multilingual for the AnimeScore axis.

For each test_v2 prompt, generate N candidates from the base model (no RL),
score each with AnimeScore and CER, then select the candidate with the highest
*gated* AnimeScore (the same zone-CER rule used by v5):

    gated(c) = a(c) + 3.0 + 0.5          if CER(c) <= 0.10  (CLEAN)
    gated(c) = a(c) + 3.0                if 0.10 < CER(c) <= 0.30  (FEASIBLE)
    gated(c) = -1.0                      if CER(c) > 0.30  (VIOLATE)

The selected candidate per prompt forms the BoN "policy" output. We report
ΔAnimeScore, CER mean/median, violation rate, and ΔUTMOS at N in {1,2,4,8},
plus the oracle (best raw AS without the gate, upper bound). Same protocol as
v5 zone-CER step 1400 evaluation: top_p=0.85, T=1.0, rep_pen=1.05, seeds
{0,100..111}.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_bon_llasa1b.py --N_max 8 --num_eval 50
"""
import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import soundfile as sf
import torch
from jiwer import cer as compute_cer
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from xcodec2.modeling_xcodec2 import XCodec2Model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--N_max", type=int, default=8)
parser.add_argument("--num_eval", type=int, default=50)
parser.add_argument("--base", default="HKUSTAudio/Llasa-1B-Multilingual")
parser.add_argument("--test_parquet", default=os.path.expanduser("~/data/llasa-tts-rl-grpo-jp-v2/test_v2.parquet"))
parser.add_argument("--output_dir", default=os.path.expanduser("~/samples/llasa_1b_bon_eval"))
args = parser.parse_args()

DEVICE = "cuda:0"
SAMPLE_RATE = 16000
SPEECH_TOKEN_START = 128264
SPEECH_TOKEN_END = 193799
SPEECH_END_ID = 128261

# Reward servers
WHISPER = os.getenv("WHISPER_SERVER", "http://localhost:8001")
ANIMESCORE = os.getenv("ANIMESCORE_SERVER", "http://localhost:8002")
UTMOS = os.getenv("UTMOS_SERVER", "http://localhost:8007")

# Zone-CER gate parameters (must match v5 training reward)
TAU_LOW = 0.10
TAU_HIGH = 0.30
RHO = 1.0
BONUS_CLEAN = 0.5
AS_OFFSET = 3.0


def gated_reward(animescore: float, cer: float) -> float:
    if cer is None or animescore is None:
        return -RHO
    if cer > TAU_HIGH:
        return -RHO
    base = max(0.0, animescore + AS_OFFSET)
    return base + BONUS_CLEAN if cer <= TAU_LOW else base


class SpeechOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids, vocab_size):
        self.mask = torch.full((vocab_size,), float("-inf"))
        for i in allowed_ids:
            self.mask[i] = 0.0

    def __call__(self, input_ids, scores):
        return scores + self.mask.to(scores.device)


def remote(url, payload, timeout=120):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [server err] {url}: {e}")
        return None


def score_cer(tokens, gt_text):
    r = remote(f"{WHISPER}/score", {"tokens": tokens, "text": gt_text})
    if not r:
        return 1.0, ""
    transcript = r.get("transcript", "")
    c = float(compute_cer(gt_text, transcript)) if transcript else 1.0
    return c, transcript


def score_as(tokens):
    r = remote(f"{ANIMESCORE}/score", {"tokens": tokens})
    return float(r["raw_score"]) if r and "raw_score" in r else None


def score_utmos(tokens):
    r = remote(f"{UTMOS}/score", {"tokens": tokens})
    if not r:
        return None
    return float(r.get("utmos") or r.get("raw_score") or r.get("score"))


def gen_one(model, tokenizer, codec, text, logits_proc, seed,
            max_new=2048, top_p=0.85, temperature=1.0, repetition_penalty=1.05):
    chat = [
        {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
    ]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, return_tensors="pt", continue_final_message=True
    ).to(DEVICE)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=SPEECH_END_ID,
            pad_token_id=tokenizer.pad_token_id or SPEECH_END_ID,
            logits_processor=[logits_proc],
        )
    new = out[0, input_ids.shape[1]:].tolist()
    if new and new[-1] == SPEECH_END_ID:
        new = new[:-1]
    ids = [t - SPEECH_TOKEN_START for t in new if SPEECH_TOKEN_START <= t <= SPEECH_TOKEN_END]
    if not ids:
        return None, []
    code_t = torch.tensor([[ids]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        wav = codec.decode_code(code_t).cpu().numpy().squeeze()
    return wav, ids


def main():
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(exist_ok=True)

    # Prompts
    df = pd.read_parquet(args.test_parquet).head(args.num_eval).reset_index(drop=True)
    prompts = [
        {"idx": int(i), "id": r["extra_info"]["id"], "text": r["extra_info"]["text"]}
        for i, r in df.iterrows()
    ]
    print(f"Loaded {len(prompts)} prompts from {args.test_parquet}")

    # Model + codec
    print(f"Loading base {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    allowed = list(range(SPEECH_TOKEN_START, SPEECH_TOKEN_END + 1)) + [SPEECH_END_ID]
    logits_proc = SpeechOnlyLogitsProcessor(allowed, model.config.vocab_size)
    print("Loading XCodec2")
    codec = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2").to(DEVICE).eval()

    # Seed pool: seed 0, then 100..(100+N_max-2). N_max=8 -> 8 seeds total per prompt.
    seeds_pool = [0] + list(range(100, 100 + args.N_max - 1))
    assert len(seeds_pool) >= args.N_max

    # Generate N_max candidates per prompt with seed-pool determinism
    all_candidates = []
    for row in prompts:
        idx = row["idx"]
        text = row["text"]
        print(f"\n[{idx:02d}] {text[:30]}…")
        cands = []
        for k, seed in enumerate(seeds_pool[: args.N_max]):
            t0 = time.time()
            wav, ids = gen_one(model, tokenizer, codec, text, logits_proc, seed=seed)
            if not ids:
                print(f"  k={k} seed={seed} EMPTY")
                continue
            cer, transcript = score_cer(ids, text)
            as_score = score_as(ids)
            utmos = score_utmos(ids)
            wav_path = cand_dir / f"{idx:03d}_k{k}.wav"
            sf.write(str(wav_path), wav, SAMPLE_RATE)
            gated = gated_reward(as_score, cer)
            cands.append({
                "k": k, "seed": seed, "wav_path": str(wav_path),
                "tokens": ids, "transcript": transcript,
                "cer": cer, "animescore": as_score, "utmos": utmos,
                "gated_reward": gated,
            })
            print(f"  k={k} seed={seed:3d} CER={cer:.3f} AS={as_score:+.2f} UTMOS={utmos:.2f} gated={gated:+.2f}  ({time.time()-t0:.1f}s)")
        all_candidates.append({"idx": idx, "id": row["id"], "text": text, "candidates": cands})

    # Free GPU
    del model, codec
    gc.collect()
    torch.cuda.empty_cache()

    # Build BoN scaling results: for each N in {1,2,4,8}, simulate BoN by
    # selecting the candidate with the highest gated reward among the first N.
    def stats_of(records):
        n = len(records)
        if n == 0:
            return None
        return {
            "n": n,
            "as_mean": float(np.mean([r["animescore"] for r in records])),
            "as_std": float(np.std([r["animescore"] for r in records])),
            "as_median": float(np.median([r["animescore"] for r in records])),
            "cer_mean": float(np.mean([r["cer"] for r in records])),
            "cer_std": float(np.std([r["cer"] for r in records])),
            "cer_median": float(np.median([r["cer"] for r in records])),
            "violation_rate": float(np.mean([1 if r["cer"] > TAU_HIGH else 0 for r in records])),
            "utmos_mean": float(np.mean([r["utmos"] for r in records])),
            "utmos_std": float(np.std([r["utmos"] for r in records])),
        }

    bon_scaling = {}
    selected_per_N = {}
    for N in [1, 2, 4, args.N_max]:
        chosen = []
        for prompt in all_candidates:
            cands = prompt["candidates"][:N]
            if not cands:
                continue
            best = max(cands, key=lambda c: c["gated_reward"])
            chosen.append({
                **{k: v for k, v in best.items() if k != "tokens"},
                "text_id": prompt["id"], "idx": prompt["idx"],
            })
        bon_scaling[N] = stats_of(chosen)
        selected_per_N[N] = chosen
        print(f"\n== BoN N={N} ==")
        print(json.dumps(bon_scaling[N], indent=2))

    # Oracle: pick the highest raw AnimeScore (no gate) from all N_max candidates
    oracle = []
    for prompt in all_candidates:
        cands = prompt["candidates"]
        if not cands:
            continue
        best = max(cands, key=lambda c: (c["animescore"] if c["animescore"] is not None else -1e9))
        oracle.append({
            **{k: v for k, v in best.items() if k != "tokens"},
            "text_id": prompt["id"], "idx": prompt["idx"],
        })
    oracle_stats = stats_of(oracle)
    print(f"\n== Oracle (max raw AS, no gate, N={args.N_max}) ==")
    print(json.dumps(oracle_stats, indent=2))

    # Save full results
    out = {
        "config": {
            "base": args.base, "num_eval": args.num_eval,
            "N_max": args.N_max, "seeds_pool": seeds_pool[: args.N_max],
            "gate": {"tau_low": TAU_LOW, "tau_high": TAU_HIGH, "rho": RHO,
                     "bonus_clean": BONUS_CLEAN, "as_offset": AS_OFFSET},
        },
        "bon_scaling": bon_scaling,
        "selected_per_N": selected_per_N,
        "oracle": oracle_stats,
        "oracle_per_prompt": oracle,
        "per_prompt_candidates": all_candidates,
    }
    out_path = out_dir / "bon_results.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
