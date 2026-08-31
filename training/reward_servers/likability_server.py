"""Lightweight REST server that returns voice likability (preference) score.

Based on CocoNut-Humoresque likability predictor (WavLM-Base+ fine-tuned).
Uses 6-class classification + expected value for wider score distribution.
Architecture: WavLM-Base+ -> mean pooling -> Linear(768, 6) -> softmax -> E[score]

Usage
-----
CUDA_VISIBLE_DEVICES=1 python reward_servers/likability_server.py --port 8003

Client POSTs JSON:  {"tokens": [123, 456, ...]}
Response:           {"raw_score": 3.45}
"""

from __future__ import annotations

import os

import argparse
from typing import List

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import WavLMModel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Likability REST server")
parser.add_argument("--port", type=int, default=8003)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2")
parser.add_argument("--ckpt", type=str,
                    default=os.environ.get("LIKABILITY_CKPT") or os.path.expanduser("~/workspace/mos-finetune-ssl/checkpoints_exp7_cls/ckpt_best"))
parser.add_argument("--ssl_name", type=str, default="microsoft/wavlm-base-plus")
args, _ = parser.parse_known_args()

DEVICE = args.device

# ---------------------------------------------------------------------------
# 1. Load XCodec2
# ---------------------------------------------------------------------------
print("Loading XCodec-2 …")
from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore
CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Likability model (MosPredictorCls - 6-class classification)
# ---------------------------------------------------------------------------
class MosPredictorCls(nn.Module):
    """Predicts a distribution over scores 1-6."""
    def __init__(self, ssl_model, ssl_out_dim, num_classes=6):
        super().__init__()
        self.ssl_model = ssl_model
        self.num_classes = num_classes
        self.output_layer = nn.Linear(ssl_out_dim, num_classes)
        self.register_buffer('score_values',
                             torch.arange(1, num_classes + 1, dtype=torch.float32))

    def forward(self, wav):
        wav = wav.squeeze(1)  # [B, T]
        res = self.ssl_model(wav)
        x = res.last_hidden_state  # [B, T', H]
        x = torch.mean(x, 1)  # [B, H]
        logits = self.output_layer(x)  # [B, num_classes]
        return logits

    def predict_score(self, wav):
        """Return expected score (float) from the predicted distribution."""
        logits = self.forward(wav)
        probs = torch.softmax(logits, dim=-1)  # [B, 6]
        expected = (probs * self.score_values.unsqueeze(0)).sum(dim=-1)  # [B]
        return expected


# ---------------------------------------------------------------------------
# 3. Build and load likability model
# ---------------------------------------------------------------------------
print(f"Loading WavLM-Base+ from {args.ssl_name} …")
ssl_model = WavLMModel.from_pretrained(args.ssl_name)

LIKABILITY = MosPredictorCls(ssl_model, ssl_model.config.hidden_size).to(DEVICE)

print(f"Loading likability checkpoint from {args.ckpt} …")
state_dict = torch.load(args.ckpt, map_location="cpu")
LIKABILITY.load_state_dict(state_dict)
LIKABILITY.eval()
for p in LIKABILITY.parameters():
    p.requires_grad = False
print("Likability model ready (6-class classification + expected value).")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Likability server", version="0.2.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token IDs")


class ScoreResponse(BaseModel):
    raw_score: float


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, None, :]
    wav = CODEC.decode_code(t)[0, 0]
    return wav.float().cpu()


@torch.inference_mode()
def compute_likability(wav: torch.Tensor) -> float:
    wav_gpu = wav.unsqueeze(0).to(DEVICE)  # [1, samples]
    s = LIKABILITY.predict_score(wav_gpu)
    return float(s.item())


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if not req.tokens:
        return ScoreResponse(raw_score=0.0)
    try:
        wav = tokens_to_wav(req.tokens)
        raw = compute_likability(wav)
        return ScoreResponse(raw_score=raw)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
