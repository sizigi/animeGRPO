"""REST server for speech naturalness (UTMOS) scoring.

Uses UTMOS22-strong (SpeechMOS) to predict mean opinion score for naturalness.
Output range: ~1-5 (MOS scale). Higher = more natural sounding.

Part of 4-axis subjective reward dynamics study.

Usage
-----
CUDA_VISIBLE_DEVICES="" python reward_servers/utmos_server.py --port 8007 --device cpu

Client POSTs JSON:  {"tokens": [123, 456, ...]}
Response:           {"utmos": 3.45}
"""

from __future__ import annotations

import argparse
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UTMOS (naturalness) REST server")
parser.add_argument("--port", type=int, default=8007)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2")
args, _ = parser.parse_known_args()

DEVICE = args.device
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# 1. Load XCodec2
# ---------------------------------------------------------------------------
print("Loading XCodec-2 ...")
from xcodec2.modeling_xcodec2 import XCodec2Model
CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Load UTMOS
# ---------------------------------------------------------------------------
print("Loading UTMOS22-strong ...")
UTMOS = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
UTMOS.eval()
for p in UTMOS.parameters():
    p.requires_grad = False
print("UTMOS model ready.")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="UTMOS naturalness server", version="0.1.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token IDs")


class ScoreResponse(BaseModel):
    utmos: float


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, None, :]
    wav = CODEC.decode_code(t)[0, 0]
    return wav.float().cpu()


@torch.inference_mode()
def compute_utmos(wav: torch.Tensor) -> float:
    wav_in = wav.unsqueeze(0)  # [1, samples]
    score = UTMOS(wav_in, SAMPLE_RATE)
    return float(score.item())


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if not req.tokens:
        return ScoreResponse(utmos=0.0)
    try:
        wav = tokens_to_wav(req.tokens)
        val = compute_utmos(wav)
        return ScoreResponse(utmos=val)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
