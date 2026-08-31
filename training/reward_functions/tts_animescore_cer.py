"""Multi-reward: Animescore (voice quality) + CER guard (intelligibility).

Animescore raw score is the primary objective.
CER acts as a guard: high CER (unintelligible) triggers a hard penalty.

Requires both servers running:
  - Animescore server (tts/animescore_server.py) on ANIMESCORE_SERVER
  - Whisper server (tts/whisper_server.py) on WHISPER_SERVER
"""

from __future__ import annotations

import os
import re
import warnings
import time
from typing import List

import requests
from jiwer import cer

# ---------------------------------------------------------------------------
# 1.  Token parsing
# ---------------------------------------------------------------------------
def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]

# ---------------------------------------------------------------------------
# 2.  Server clients
# ---------------------------------------------------------------------------
ANIMESCORE_SERVER = os.getenv("ANIMESCORE_SERVER", "http://localhost:8002")
ANIMESCORE_SCORE_URL = f"{ANIMESCORE_SERVER.rstrip('/')}/score"
ANIMESCORE_HEALTH_URL = f"{ANIMESCORE_SERVER.rstrip('/')}/healthz"

WHISPER_SERVER = os.getenv("WHISPER_SERVER", "http://localhost:8001")
WHISPER_SCORE_URL = f"{WHISPER_SERVER.rstrip('/')}/score"
WHISPER_HEALTH_URL = f"{WHISPER_SERVER.rstrip('/')}/healthz"

_last_health_animescore = 0.0
_last_health_whisper = 0.0


def _check_animescore():
    global _last_health_animescore
    if time.time() - _last_health_animescore < 30:
        return
    try:
        requests.get(ANIMESCORE_HEALTH_URL, timeout=10)
        _last_health_animescore = time.time()
    except Exception as e:
        raise RuntimeError(f"Animescore server not reachable at {ANIMESCORE_SERVER}: {e}")


def _check_whisper():
    global _last_health_whisper
    if time.time() - _last_health_whisper < 30:
        return
    try:
        requests.get(WHISPER_HEALTH_URL, timeout=10)
        _last_health_whisper = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {WHISPER_SERVER}: {e}")


def _remote_animescore(tokens: List[int]) -> dict:
    _check_animescore()
    r = requests.post(ANIMESCORE_SCORE_URL, json={"tokens": tokens}, timeout=120)
    r.raise_for_status()
    return r.json()


def _remote_whisper(tokens: List[int], text: str) -> dict:
    _check_whisper()
    r = requests.post(WHISPER_SCORE_URL, json={"tokens": tokens, "text": text}, timeout=120)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# 3.  Reward computation
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    animescore_weight: float = 0.6,
    cer_weight: float = 0.4,
) -> float:
    """Composite reward: weighted animescore + soft CER penalty.

    R = animescore_weight * raw_animescore + cer_weight * max(0, 1 - 2*CER)

    CER contribution is continuous:
      CER=0.0 → +0.4, CER=0.25 → +0.2, CER=0.5 → 0.0, CER>0.5 → 0.0
    No hard penalty — samples with CER>0.5 still get animescore gradient.
    """
    ids = _parse_ids(solution_str)

    if not ids:
        print(f"\033[91mAnimescore+CER: no tokens, Reward: 0.0000\033[0m")
        return 0.0

    # --- Animescore ---
    try:
        resp_anim = _remote_animescore(ids)
        raw_animescore = float(resp_anim["raw_score"])
    except Exception as e:
        warnings.warn(f"Animescore server error: {e}; returning 0")
        print(f"\033[91mAnimescore+CER: animescore ERROR, Reward: 0.0000\033[0m")
        return 0.0

    # --- CER ---
    try:
        resp_whisper = _remote_whisper(ids, ground_truth)
        transcript = resp_whisper.get("transcript", "")
        c = float(cer(ground_truth, transcript)) if transcript else 1.0
    except Exception as e:
        warnings.warn(f"Whisper server error: {e}; assuming CER=0 (no penalty)")
        c = 0.0

    # --- Soft composite reward ---
    r_style = raw_animescore
    r_cer = max(0.0, 1.0 - 2.0 * c)  # CER=0→1.0, CER=0.5→0.0, CER>0.5→0.0
    reward = animescore_weight * r_style + cer_weight * r_cer

    print(
        f"\033[94mAnimescore: {raw_animescore:.4f}, CER: {c:.3f}, "
        f"r_cer: {r_cer:.3f}, Reward: {reward:.4f}\033[0m"
    )
    return reward


# CLI quick test
if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(
        {"reward": compute_score("cli", sys.argv[1], sys.argv[2])},
        indent=2, ensure_ascii=False,
    ))
