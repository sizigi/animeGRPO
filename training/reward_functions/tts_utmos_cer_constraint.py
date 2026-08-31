"""Constrained reward: UTMOS objective with CER as a hard constraint.

Mirror of tts_animescore_cer_constraint.py with AnimeScore swapped for UTMOS.
Only the reward axis changes; zone structure (VIOLATE/FEASIBLE/CLEAN), CER thresholds,
violation penalty, low-CER bonus all identical to v5.

UTMOS raw range ~[1, 5] (MOS scale, already non-negative), so no offset needed —
unlike AnimeScore which required +3 shift to clear the violation penalty.

Returns dict for verl per-axis logging:
  {"score", "utmos", "cer", "cer_violation", "ok"}

Requires:
  - UTMOS server (reward_servers/utmos_server.py) on UTMOS_SERVER (:8007)
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


UTMOS_SERVER   = os.getenv("UTMOS_SERVER",   "http://localhost:8007")
WHISPER_SERVER = os.getenv("WHISPER_SERVER", "http://localhost:8001")

UTMOS_SCORE_URL    = f"{UTMOS_SERVER.rstrip('/')}/score"
UTMOS_HEALTH_URL   = f"{UTMOS_SERVER.rstrip('/')}/healthz"
WHISPER_SCORE_URL  = f"{WHISPER_SERVER.rstrip('/')}/score"
WHISPER_HEALTH_URL = f"{WHISPER_SERVER.rstrip('/')}/healthz"

_last_utmos = 0.0
_last_whisper = 0.0


def _check_utmos():
    global _last_utmos
    if time.time() - _last_utmos < 30:
        return
    try:
        requests.get(UTMOS_HEALTH_URL, timeout=10)
        _last_utmos = time.time()
    except Exception as e:
        raise RuntimeError(f"UTMOS server not reachable at {UTMOS_SERVER}: {e}")


def _check_whisper():
    global _last_whisper
    if time.time() - _last_whisper < 30:
        return
    try:
        requests.get(WHISPER_HEALTH_URL, timeout=10)
        _last_whisper = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {WHISPER_SERVER}: {e}")


def _remote_utmos(tokens: List[int]) -> dict:
    _check_utmos()
    r = requests.post(UTMOS_SCORE_URL, json={"tokens": tokens}, timeout=180)
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
    """Constrained reward with UTMOS objective.

    UTMOS is in [1, 5] (MOS scale), already non-negative, so we use it directly.

    Zones:
      Zone 1 (CLEAN):    CER ≤ cer_low      → R = utmos + low_cer_bonus
      Zone 2 (FEASIBLE): cer_low < CER ≤ cer_high → R = utmos
      Zone 3 (VIOLATE):  CER > cer_high     → R = violation_penalty (= -1.0)

    Ordering invariant: violation_penalty (-1.0) < utmos_min (~1.0) < ... < clean
    """
    ids = _parse_ids(solution_str)

    zero = {"score": 0.0, "utmos": 0.0, "cer": 1.0,
            "cer_violation": 1.0, "ok": 0.0}

    if not ids:
        print("\033[91mUTMOS+CER-constraint: no tokens, R=0.0\033[0m")
        return zero

    try:
        resp = _remote_utmos(ids)
        utmos = float(resp["utmos"])
    except Exception as e:
        warnings.warn(f"UTMOS server error: {e}; R=0")
        print("\033[91mUTMOS+CER-constraint: utmos ERROR, R=0.0\033[0m")
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
        reward = utmos + low_cer_bonus
        violation = 0.0
        zone = "CLEAN"
        color = "\033[92m"  # green
    else:
        reward = utmos
        violation = 0.0
        zone = "FEASIBLE"
        color = "\033[93m"  # yellow

    print(
        f"{color}UTMOS={utmos:.3f} CER={c:.3f} [{zone}] -> R={reward:+.4f}\033[0m"
    )
    return {
        "score":         reward,
        "utmos":         utmos,
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
