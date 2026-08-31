"""Lightweight REST server that returns anime voice quality score.

Usage
-----
CUDA_VISIBLE_DEVICES=1 python tts/animescore_server.py --port 8002 --ckpt /path/to/ckpt.pt

Client POSTs JSON:  {"tokens": [123, 456, ...]}
Response:           {"raw_score": 2.345}
"""

from __future__ import annotations

import os

import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Animescore REST server")
parser.add_argument("--port", type=int, default=8002)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--codec", type=str, default="HKUST-Audio/xcodec2")
parser.add_argument("--ckpt", type=str,
                    default=os.environ.get("ANIMESCORE_CKPT") or os.path.expanduser("~/workspace/mos-finetune-ssl/output/ckpts/ckpt_wavlm_frozen_best.pt"))
parser.add_argument("--ssl_type", type=str, default="wavlm")
parser.add_argument("--ssl_name", type=str, default="microsoft/wavlm-base")
parser.add_argument("--layer_mixing_k", type=int, default=4)
args, _ = parser.parse_known_args()

DEVICE = args.device

# ---------------------------------------------------------------------------
# 1. Load XCodec2
# ---------------------------------------------------------------------------
print("Loading XCodec-2 …")
from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore
CODEC = XCodec2Model.from_pretrained(args.codec).to(DEVICE).eval()

# ---------------------------------------------------------------------------
# 2. Animescore model definition (from mos-finetune-ssl, self-contained)
# ---------------------------------------------------------------------------
@dataclass
class SSLSpec:
    ssl_type: str
    name_or_path: str
    target_sr: int
    feat_dim: int


class HFSSLWrapper(nn.Module):
    def __init__(self, name_or_path: str, output_hidden_states: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(name_or_path, output_hidden_states=output_hidden_states)
        self.model = AutoModel.from_pretrained(name_or_path, config=self.config)
        self.output_hidden_states = output_hidden_states
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(name_or_path)
        self.target_sr = getattr(self.feature_extractor, "sampling_rate", 16000)
        self.feat_dim = int(self.config.hidden_size)

    def forward(self, wav_16k: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out = self.model(input_values=wav_16k, attention_mask=attention_mask)
        return {
            "last_hidden_state": out.last_hidden_state,
            "hidden_states": getattr(out, "hidden_states", None),
        }


class LayerMixing(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.alpha, dim=0)
        out = 0.0
        for i, h in enumerate(hidden_states):
            out = out + w[i] * h
        return out


class RankNetMos(nn.Module):
    def __init__(self, ssl: nn.Module, use_layer_mixing_last_k: int = 4,
                 lstm_hidden: int = 256, lstm_layers: int = 1,
                 mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ssl = ssl
        self.use_layer_mixing_last_k = int(use_layer_mixing_last_k)
        feat_dim = getattr(self.ssl, "feat_dim", None)
        if feat_dim is None:
            raise RuntimeError("ssl wrapper must expose feat_dim")

        self.layer_mixer = None
        if self.use_layer_mixing_last_k > 1:
            self.layer_mixer = LayerMixing(self.use_layer_mixing_last_k)

        self.bilstm = nn.LSTM(input_size=feat_dim, hidden_size=lstm_hidden,
                              num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(2 * lstm_hidden), nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def extract_feats(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.ssl(wav)
        hs = out.get("hidden_states", None)
        if self.layer_mixer is not None and hs is not None:
            last_k = hs[-self.use_layer_mixing_last_k:]
            feats = self.layer_mixer(list(last_k))
        else:
            feats = out["last_hidden_state"]
        return feats

    def score(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feats(wav)
        z, _ = self.bilstm(feats)
        zbar = z.mean(dim=1)
        s = self.mlp(zbar).squeeze(-1)
        return s


def load_ckpt(model: nn.Module, ckpt_path: str, map_location="cpu"):
    sd = torch.load(ckpt_path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    fixed = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    matched = len(set(fixed.keys()) & set(model.state_dict().keys()))
    print(f"[CKPT] keys: {len(fixed)} | model: {len(model.state_dict())} | matched: {matched}")
    if missing:
        print(f"[CKPT] missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"[CKPT] unexpected (first 10): {unexpected[:10]}")


# ---------------------------------------------------------------------------
# 3. Build and load animescore model
# ---------------------------------------------------------------------------
print("Loading Animescore model …")
ssl = HFSSLWrapper(args.ssl_name, output_hidden_states=True).to(DEVICE)
for p in ssl.parameters():
    p.requires_grad = False

ANIMESCORE = RankNetMos(ssl=ssl, use_layer_mixing_last_k=args.layer_mixing_k).to(DEVICE)
load_ckpt(ANIMESCORE, args.ckpt, map_location="cpu")
ANIMESCORE.eval()
for p in ANIMESCORE.parameters():
    p.requires_grad = False
print(f"Animescore ready: ssl={args.ssl_type}, ckpt={args.ckpt}")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(title="Animescore server", version="0.1.0")


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
def compute_animescore(wav: torch.Tensor) -> float:
    wav_gpu = wav.unsqueeze(0).to(DEVICE)  # [1, samples]
    s = ANIMESCORE.score(wav_gpu)
    return float(s.item())


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    if not req.tokens:
        return ScoreResponse(raw_score=0.0)
    try:
        wav = tokens_to_wav(req.tokens)
        raw = compute_animescore(wav)
        return ScoreResponse(raw_score=raw)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
