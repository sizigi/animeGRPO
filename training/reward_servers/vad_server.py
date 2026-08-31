"""REST server for emotional intensity (VAD) scoring.

Uses audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to predict
arousal, dominance, valence from speech. Returns arousal as the primary
"emotional intensity" reward signal for GRPO.

Output range: ~0-1 for each dimension (arousal, dominance, valence).

Usage
-----
CUDA_VISIBLE_DEVICES=1 python reward_servers/vad_server.py --port 8006

Client POSTs JSON:  {"tokens": [123, 456, ...]}
Response:           {"arousal": 0.62, "dominance": 0.45, "valence": 0.51}
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="VAD (Arousal/Dominance/Valence) REST server")
parser.add_argument("--port", type=int, default=8006)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2")
parser.add_argument("--vad_model", type=str,
                    default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
args, _ = parser.parse_known_args()

DEVICE = args.device

# ---------------------------------------------------------------------------
# 1. Load XCodec2
# ---------------------------------------------------------------------------
print("Loading XCodec-2 ...")
from xcodec2.modeling_xcodec2 import XCodec2Model
CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Custom EmotionModel (audeering architecture)
# ---------------------------------------------------------------------------
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


# ---------------------------------------------------------------------------
# 3. Load VAD model
# ---------------------------------------------------------------------------
print(f"Loading VAD model: {args.vad_model} ...")
PROCESSOR = Wav2Vec2Processor.from_pretrained(args.vad_model)
VAD_MODEL = EmotionModel.from_pretrained(args.vad_model).to(DEVICE).eval()
for p in VAD_MODEL.parameters():
    p.requires_grad = False
print("VAD model ready (arousal/dominance/valence regression).")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="VAD Emotion server", version="0.1.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token IDs")


class ScoreResponse(BaseModel):
    arousal: float
    dominance: float
    valence: float


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    t = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, None, :]
    wav = CODEC.decode_code(t)[0, 0]
    return wav.float().cpu()


@torch.inference_mode()
def compute_vad(wav: torch.Tensor) -> dict:
    wav_np = wav.numpy()
    inputs = PROCESSOR(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)
    _, logits = VAD_MODEL(input_values)
    # logits: [1, 3] -> arousal, dominance, valence
    vals = logits[0].cpu().numpy()
    return {
        "arousal": float(vals[0]),
        "dominance": float(vals[1]),
        "valence": float(vals[2]),
    }


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if not req.tokens:
        return ScoreResponse(arousal=0.0, dominance=0.0, valence=0.0)
    try:
        wav = tokens_to_wav(req.tokens)
        result = compute_vad(wav)
        return ScoreResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
