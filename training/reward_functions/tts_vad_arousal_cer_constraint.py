"""Constrained reward: VAD Arousal objective with CER as a hard constraint.

Mirror of tts_utmos_cer_constraint.py with UTMOS swapped for VAD arousal.
Only the reward axis changes; zone structure (VIOLATE/FEASIBLE/CLEAN), CER
thresholds, violation_penalty, low_cer_bonus all identical.

VAD arousal raw range ~[0, 1] (audeering wav2vec2, already non-negative),
so no offset needed. Note that the reward magnitude is smaller than UTMOS
(max ~1.5 vs ~5.5 for UTMOS), so gradient steps may be more conservative —
consistent with the user's intent to ablate ONLY the reward axis.

Returns dict for verl per-axis logging:
  {"score", "arousal", "dominance", "valence", "cer", "cer_violation", "ok"}

Requires:
  - VAD server (reward_servers/vad_server.py) on VAD_SERVER (:8006)
  - Whisper server on WHISPER_SERVER (:8001)
"""

from __future__ import annotations

import os
import re
import time
import warnings
from typing import List

import requests
from jiwer import cer


def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]


VAD_SERVER     = os.getenv("VAD_SERVER",     "http://localhost:8006")
WHISPER_SERVER = os.getenv("WHISPER_SERVER", "http://localhost:8001")

VAD_SCORE_URL      = f"{VAD_SERVER.rstrip('/')}/score"
VAD_HEALTH_URL     = f"{VAD_SERVER.rstrip('/')}/healthz"
WHISPER_SCORE_URL  = f"{WHISPER_SERVER.rstrip('/')}/score"
WHISPER_HEALTH_URL = f"{WHISPER_SERVER.rstrip('/')}/healthz"

_last_vad = 0.0
_last_whisper = 0.0


def _check_vad():
    global _last_vad
    if time.time() - _last_vad < 30:
        return
    try:
        requests.get(VAD_HEALTH_URL, timeout=10)
        _last_vad = time.time()
    except Exception as e:
        raise RuntimeError(f"VAD server not reachable at {VAD_SERVER}: {e}")


def _check_whisper():
    global _last_whisper
    if time.time() - _last_whisper < 30:
        return
    try:
        requests.get(WHISPER_HEALTH_URL, timeout=10)
        _last_whisper = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {WHISPER_SERVER}: {e}")


def _remote_vad(tokens: List[int]) -> dict:
    _check_vad()
    r = requests.post(VAD_SCORE_URL, json={"tokens": tokens}, timeout=180)
    r.raise_for_status()
    return r.json()


def _remote_whisper(tokens: List[int], text: str) -> dict:
    _check_whisper()
    r = requests.post(WHISPER_SCORE_URL, json={"tokens": tokens, "text": text}, timeout=180)
    r.raise_for_status()
    return r.json()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    *,
    cer_low: float = 0.10,
    cer_high: float = 0.30,
    violation_penalty: float = -1.0,
    low_cer_bonus: float = 0.5,
) -> dict:
    """Constrained reward with VAD Arousal objective.

    Arousal in [0, 1], used directly. Same zone structure as v5:
      Zone 1 (CLEAN):    CER ≤ cer_low      → R = arousal + low_cer_bonus
      Zone 2 (FEASIBLE): cer_low < CER ≤ cer_high → R = arousal
      Zone 3 (VIOLATE):  CER > cer_high     → R = violation_penalty (= -1.0)
    """
    ids = _parse_ids(solution_str)

    zero = {"score": 0.0, "arousal": 0.0, "dominance": 0.0, "valence": 0.0,
            "cer": 1.0, "cer_violation": 1.0, "ok": 0.0}

    if not ids:
        print("\033[91mVAD-arousal+CER-constraint: no tokens, R=0.0\033[0m")
        return zero

    try:
        resp = _remote_vad(ids)
        arousal   = float(resp["arousal"])
        dominance = float(resp.get("dominance", 0.0))
        valence   = float(resp.get("valence", 0.0))
        # clamp to [0,1] in case of slight model overshoot
        arousal_clip = max(0.0, min(1.0, arousal))
    except Exception as e:
        warnings.warn(f"VAD server error: {e}; R=0")
        print("\033[91mVAD+CER-constraint: vad ERROR, R=0.0\033[0m")
        return zero

    try:
        resp_w = _remote_whisper(ids, ground_truth)
        transcript = resp_w.get("transcript", "")
        c = float(cer(ground_truth, transcript)) if transcript else 1.0
    except Exception as e:
        warnings.warn(f"Whisper server error: {e}; assuming CER=1 (violation)")
        c = 1.0

    if c > cer_high:
        reward = violation_penalty
        violation = 1.0
        zone = "VIOLATE"
        color = "\033[91m"  # red
    elif c <= cer_low:
        reward = arousal_clip + low_cer_bonus
        violation = 0.0
        zone = "CLEAN"
        color = "\033[92m"  # green
    else:
        reward = arousal_clip
        violation = 0.0
        zone = "FEASIBLE"
        color = "\033[93m"  # yellow

    print(
        f"{color}Arousal={arousal:.3f} (D={dominance:.3f} V={valence:.3f}) "
        f"CER={c:.3f} [{zone}] -> R={reward:+.4f}\033[0m"
    )
    return {
        "score":         reward,
        "arousal":       arousal,
        "dominance":     dominance,
        "valence":       valence,
        "cer":           c,
        "cer_violation": violation,
        "ok":            1.0,
    }


if __name__ == "__main__":
    import sys, json
    print(json.dumps(
        compute_score("cli", sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""),
        indent=2, ensure_ascii=False,
    ))
