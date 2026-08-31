"""Evaluate 1B-Multilingual v5 (AnimeScore+CER constrained) vs base on test_v2.

Models:
  - base : HKUSTAudio/Llasa-1B-Multilingual           (untrained)
  - v5   : llasa-1b-v5-step510-merged                  (best by val_R 1.057)

Sampling: nucleus (top_p=0.85, temp=1.0, repetition_penalty=1.05) — matches training.
1 seed (=0).

Metrics (per sentence):
  - animescore    (animescore server, port 8002)
  - CER, transcript (whisper server, port 8001)
  - UTMOS         (local SpeechMOS, tarepan/SpeechMOS)
  - arousal, dominance, valence (VAD server, port 8006)
  - likability    (likability server, port 8003)
  - speaker_sim   (speaker_sim server, port 8004)
  - MCD           (pymcd, ref = base output per sentence)

Outputs:
  ~/samples/llasa_1b_v5_eval/eval_results.json   (per-sample raw)
  ~/samples/llasa_1b_v5_eval/aggregate.json      (per-model mean/std)
  ~/samples/llasa_1b_v5_eval/<model>/{000..049}.wav
"""

import os
import sys
import json
import time
import gc
import warnings
import re

import torch
import numpy as np
import soundfile as sf
import pandas as pd
import requests
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor
from xcodec2.modeling_xcodec2 import XCodec2Model
from jiwer import cer as compute_cer

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(os.path.expanduser("~/samples/llasa_1b_v5_step1400_eval"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_PARQUET = os.path.expanduser("~/data/llasa-tts-rl-grpo-jp-v2/test_v2.parquet")

SAMPLE_RATE = 16000
NUM_EVAL = 50
SEED = 0
DEVICE = "cuda:0"

ANIMESCORE_SERVER  = os.getenv("ANIMESCORE_SERVER",  "http://localhost:8002")
WHISPER_SERVER     = os.getenv("WHISPER_SERVER",     "http://localhost:8001")
VAD_SERVER         = os.getenv("VAD_SERVER",         "http://localhost:8006")
LIKABILITY_SERVER  = os.getenv("LIKABILITY_SERVER",  "http://localhost:8003")
SPEAKER_SIM_SERVER = os.getenv("SPEAKER_SIM_SERVER", "http://localhost:8004")

MODELS = {
    "base":     "HKUSTAudio/Llasa-1B-Multilingual",
    "v5_1400":  os.path.expanduser("~/checkpoints/llasa-1b-v5-step1400-merged"),
}

SPEECH_TOKEN_START = 128264
SPEECH_TOKEN_END   = 193799
SPEECH_END_ID      = 128261  # SPEECH_GENERATION_END


class SpeechOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids, vocab_size):
        self.mask = torch.full((vocab_size,), float("-inf"))
        for i in allowed_ids:
            self.mask[i] = 0.0

    def __call__(self, input_ids, scores):
        return scores + self.mask.to(scores.device)


def parse_speech_ids(decoded: str):
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", decoded)]


def load_test_data():
    df = pd.read_parquet(TEST_PARQUET).head(NUM_EVAL).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        text = r["extra_info"]["text"]
        sid = r["extra_info"]["id"]
        rows.append({"idx": int(i), "id": sid, "text": text})
    return rows


def generate_speech(model, tokenizer, codec, text, logits_proc, max_new=2048,
                    top_p=0.85, temperature=1.0, repetition_penalty=1.05, seed=0):
    chat = [
        {"role": "user",
         "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"},
        {"role": "assistant",
         "content": "<|SPEECH_GENERATION_START|>"},
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
            max_new_tokens=max_new,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=SPEECH_END_ID,
            pad_token_id=tokenizer.pad_token_id or SPEECH_END_ID,
            logits_processor=[logits_proc],
        )

    new_tokens = out[0, input_ids.shape[1]:].tolist()
    if new_tokens and new_tokens[-1] == SPEECH_END_ID:
        new_tokens = new_tokens[:-1]
    code_ids = [t - SPEECH_TOKEN_START for t in new_tokens
                if SPEECH_TOKEN_START <= t <= SPEECH_TOKEN_END]
    if not code_ids:
        return None, []

    code_t = torch.tensor([[code_ids]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        wav = codec.decode_code(code_t).cpu().numpy().squeeze()
    return wav, code_ids


def remote(url, payload, timeout=180):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [server error] {url}: {e}")
        return None


def score_all(tokens, gt_text, wav, sr, base_wav=None):
    out = {}
    # AnimeScore
    r = remote(f"{ANIMESCORE_SERVER}/score", {"tokens": tokens})
    out["animescore"] = float(r["raw_score"]) if r else None
    # CER (Whisper)
    r = remote(f"{WHISPER_SERVER}/score", {"tokens": tokens, "text": gt_text})
    if r:
        out["transcript"] = r.get("transcript", "")
        c = float(compute_cer(gt_text, out["transcript"])) if out["transcript"] else 1.0
        out["cer"] = c
    else:
        out["transcript"] = ""
        out["cer"] = 1.0
    # VAD
    r = remote(f"{VAD_SERVER}/score", {"tokens": tokens})
    if r:
        out["arousal"]   = float(r.get("arousal", 0.0))
        out["dominance"] = float(r.get("dominance", 0.0))
        out["valence"]   = float(r.get("valence", 0.0))
    # Likability — server returns {"raw_score": ...}
    r = remote(f"{LIKABILITY_SERVER}/score", {"tokens": tokens})
    if r is not None:
        out["likability"] = float(r.get("raw_score", 0.0))
    # Speaker-Sim — uses JSUT reference internally; returns {"raw_score": ...}
    r = remote(f"{SPEAKER_SIM_SERVER}/score", {"tokens": tokens})
    if r is not None:
        out["speaker_sim"] = float(r.get("raw_score", 0.0))
    return out


def utmos_local(wav, sr, predictor):
    if predictor is None:
        return None
    import torch as _t
    t = _t.tensor(wav, dtype=_t.float32).unsqueeze(0)
    if sr != 16000:
        import torchaudio.functional as AF
        t = AF.resample(t, sr, 16000)
    with _t.no_grad():
        score = predictor(t.to(DEVICE), 16000)
    return float(score.cpu().item())


def main():
    print(f"Loading test data: {TEST_PARQUET}")
    rows = load_test_data()
    print(f"  {len(rows)} samples")

    print("Loading XCodec2...")
    codec = XCodec2Model.from_pretrained("HKUST-Audio/xcodec2").to(DEVICE).eval()

    print("Loading UTMOS (local SpeechMOS)...")
    try:
        utmos_pred = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                     trust_repo=True).to(DEVICE).eval()
    except Exception as e:
        print(f"  UTMOS load failed: {e}, skipping")
        utmos_pred = None

    all_results = []
    base_wav_cache = {}  # for speaker_sim ref

    for model_name, model_path in MODELS.items():
        print(f"\n{'='*70}\nModel: {model_name}  ({model_path})")
        print(f"{'='*70}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to(DEVICE).eval()

        allowed = list(range(SPEECH_TOKEN_START, SPEECH_TOKEN_END + 1)) + [SPEECH_END_ID]
        logits_proc = SpeechOnlyLogitsProcessor(allowed, model.config.vocab_size)

        out_dir = OUTPUT_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            idx, text, sid = row["idx"], row["text"], row["id"]
            t0 = time.time()
            wav, ids = generate_speech(model, tokenizer, codec, text, logits_proc, seed=SEED)
            if wav is None or not ids:
                print(f"  [{idx:02d}] EMPTY generation, skip")
                all_results.append({"model": model_name, "idx": idx, "id": sid, "text": text,
                                    "ok": False})
                continue

            wav_path = out_dir / f"{idx:03d}.wav"
            sf.write(str(wav_path), wav, SAMPLE_RATE)

            # Cache base wav for speaker_sim ref
            if model_name == "base":
                base_wav_cache[idx] = wav
            base_ref = base_wav_cache.get(idx) if model_name != "base" else None

            metrics = score_all(ids, text, wav, SAMPLE_RATE, base_wav=base_ref)
            metrics["utmos"] = utmos_local(wav, SAMPLE_RATE, utmos_pred)
            metrics["duration_s"] = round(len(wav) / SAMPLE_RATE, 3)

            row_out = {"model": model_name, "idx": idx, "id": sid, "text": text,
                       "wav_path": str(wav_path), "ok": True, **metrics}
            all_results.append(row_out)

            cer_v = metrics.get("cer", 1.0)
            as_v  = metrics.get("animescore", 0.0)
            ut_v  = metrics.get("utmos") or 0.0
            print(f"  [{idx:02d}/{len(rows)}] {sid}  AS={as_v:+.3f} CER={cer_v:.3f} "
                  f"UTMOS={ut_v:.3f} ({time.time()-t0:.1f}s)")

        # Write incremental results
        json.dump(all_results, open(OUTPUT_DIR / "eval_results.json", "w"),
                  indent=2, ensure_ascii=False)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate per-model
    print("\nAggregating...")
    agg = {}
    for m in MODELS.keys():
        rows_m = [r for r in all_results if r["model"] == m and r.get("ok")]
        d = {}
        for k in ("animescore", "cer", "utmos", "arousal", "dominance", "valence",
                  "likability", "speaker_sim", "duration_s"):
            vals = [r[k] for r in rows_m if r.get(k) is not None]
            if vals:
                d[f"{k}_mean"] = float(np.mean(vals))
                d[f"{k}_std"]  = float(np.std(vals))
                d[f"{k}_n"]    = len(vals)
        d["n_total"]   = len([r for r in all_results if r["model"] == m])
        d["n_ok"]      = len(rows_m)
        agg[m] = d

    json.dump(agg, open(OUTPUT_DIR / "aggregate.json", "w"), indent=2, ensure_ascii=False)
    print(f"  -> {OUTPUT_DIR}/aggregate.json")
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
