"""Constrained reward: Likability objective with CER as a hard constraint.

Mirror of tts_utmos_cer_constraint.py with UTMOS swapped for Likability
(CocoNut-Humoresque WavLM-Base+ predictor). Only the reward axis changes.

Likability raw range ~[1, 5] (preference MOS-style), already non-negative,
so no offset needed — same shape as UTMOS reward.

Returns dict for verl per-axis logging:
  {"score", "likability", "cer", "cer_violation", "ok"}

Requires:
  - Likability server (reward_servers/likability_server.py) on LIKABILITY_SERVER (:8003)
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


LIKABILITY_SERVER = os.getenv("LIKABILITY_SERVER", "http://localhost:8003")
WHISPER_SERVER    = os.getenv("WHISPER_SERVER",    "http://localhost:8001")

LIKABILITY_SCORE_URL  = f"{LIKABILITY_SERVER.rstrip('/')}/score"
LIKABILITY_HEALTH_URL = f"{LIKABILITY_SERVER.rstrip('/')}/healthz"
WHISPER_SCORE_URL     = f"{WHISPER_SERVER.rstrip('/')}/score"
WHISPER_HEALTH_URL    = f"{WHISPER_SERVER.rstrip('/')}/healthz"

_last_likability = 0.0
_last_whisper = 0.0


def _check_likability():
    global _last_likability
    if time.time() - _last_likability < 30:
        return
    try:
        requests.get(LIKABILITY_HEALTH_URL, timeout=10)
        _last_likability = time.time()
    except Exception as e:
        raise RuntimeError(f"Likability server not reachable at {LIKABILITY_SERVER}: {e}")


def _check_whisper():
    global _last_whisper
    if time.time() - _last_whisper < 30:
        return
    try:
        requests.get(WHISPER_HEALTH_URL, timeout=10)
        _last_whisper = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {WHISPER_SERVER}: {e}")


def _remote_likability(tokens: List[int]) -> dict:
    _check_likability()
    r = requests.post(LIKABILITY_SCORE_URL, json={"tokens": tokens}, timeout=180)
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
    """Constrained reward with Likability objective.

    Likability in [1, 5], used directly. Same zone structure as v5:
      Zone 1 (CLEAN):    CER ≤ cer_low      → R = likability + low_cer_bonus
      Zone 2 (FEASIBLE): cer_low < CER ≤ cer_high → R = likability
      Zone 3 (VIOLATE):  CER > cer_high     → R = violation_penalty (= -1.0)
    """
    ids = _parse_ids(solution_str)

    zero = {"score": 0.0, "likability": 0.0, "cer": 1.0,
            "cer_violation": 1.0, "ok": 0.0}

    if not ids:
        print("\033[91mLikability+CER-constraint: no tokens, R=0.0\033[0m")
        return zero

    try:
        resp = _remote_likability(ids)
        likability = float(resp["raw_score"])
    except Exception as e:
        warnings.warn(f"Likability server error: {e}; R=0")
        print("\033[91mLikability+CER-constraint: lik ERROR, R=0.0\033[0m")
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
        reward = likability + low_cer_bonus
        violation = 0.0
        zone = "CLEAN"
        color = "\033[92m"  # green
    else:
        reward = likability
        violation = 0.0
        zone = "FEASIBLE"
        color = "\033[93m"  # yellow

    print(
        f"{color}Likability={likability:.3f} CER={c:.3f} [{zone}] -> R={reward:+.4f}\033[0m"
    )
    return {
        "score":         reward,
        "likability":    likability,
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
