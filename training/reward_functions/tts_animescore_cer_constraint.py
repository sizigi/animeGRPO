"""Constrained reward: AnimeScore objective with CER as a hard constraint.

Different from tts_animescore_cer.py (weighted-sum). Here CER is a *gate*:
  - CER ≤ cer_low (default 0.10)        → R = animescore + bonus
  - cer_low < CER ≤ cer_high (0.30)     → R = animescore (in feasible zone)
  - CER > cer_high                      → R = penalty  (constraint violated)

Why a hard constraint instead of a weighted sum:
  - Weighted sum (R = α·anime + β·(1−2·CER)) creates a Pareto trade-off where
    the model can sacrifice intelligibility to maximize anime style. With a
    hard cap the gradient on animescore disappears for unintelligible outputs
    so the model is pushed to first satisfy CER, then maximize style.

Returns a dict so verl logs per-axis to wandb:
  {"score", "animescore", "cer", "cer_violation", "ok"}

Requires:
  - Animescore server (reward_servers/alignment_server.py or animescore_server.py) on ANIMESCORE_SERVER (:8002 or :8005)
  - Whisper server (reward_servers/whisper_server.py) on WHISPER_SERVER (:8001)
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


ANIMESCORE_SERVER = os.getenv("ANIMESCORE_SERVER", "http://localhost:8002")
WHISPER_SERVER    = os.getenv("WHISPER_SERVER",    "http://localhost:8001")

ANIMESCORE_SCORE_URL  = f"{ANIMESCORE_SERVER.rstrip('/')}/score"
ANIMESCORE_HEALTH_URL = f"{ANIMESCORE_SERVER.rstrip('/')}/healthz"
WHISPER_SCORE_URL     = f"{WHISPER_SERVER.rstrip('/')}/score"
WHISPER_HEALTH_URL    = f"{WHISPER_SERVER.rstrip('/')}/healthz"

_last_anime = 0.0
_last_whisper = 0.0


def _check_anime():
    global _last_anime
    if time.time() - _last_anime < 30:
        return
    try:
        requests.get(ANIMESCORE_HEALTH_URL, timeout=10)
        _last_anime = time.time()
    except Exception as e:
        raise RuntimeError(f"Animescore server not reachable at {ANIMESCORE_SERVER}: {e}")


def _check_whisper():
    global _last_whisper
    if time.time() - _last_whisper < 30:
        return
    try:
        requests.get(WHISPER_HEALTH_URL, timeout=10)
        _last_whisper = time.time()
    except Exception as e:
        raise RuntimeError(f"Whisper server not reachable at {WHISPER_SERVER}: {e}")


def _remote_anime(tokens: List[int]) -> dict:
    _check_anime()
    r = requests.post(ANIMESCORE_SCORE_URL, json={"tokens": tokens}, timeout=180)
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
    animescore_offset: float = 3.0,    # raw range ~[-3, +5] → [0, +8]
    violation_penalty: float = -1.0,
    low_cer_bonus: float = 0.5,
) -> dict:
    """Constrained reward with strict zone ordering.

    The 1B-Multilingual base produces raw animescore in roughly [-3, +5].
    To keep `feasible reward > violation penalty` strictly, we shift+clip
    animescore into a non-negative range first:

        as_pos = max(0, animescore + animescore_offset)

    Zones:
      Zone 1 (feasible-clean): CER ≤ cer_low
          R = as_pos + low_cer_bonus      (≥ 0.5)
      Zone 2 (feasible):       cer_low < CER ≤ cer_high
          R = as_pos                       (≥ 0.0)
      Zone 3 (infeasible):     CER > cer_high
          R = violation_penalty           (= -1.0)

    Ordering invariant: violation < feasible ≤ clean for all observed
    animescores. This prevents the policy from finding a "fail-cheaply"
    mode where breaking the output yields a higher reward than passing
    intelligibility but with a low animescore.
    """
    ids = _parse_ids(solution_str)

    zero = {"score": 0.0, "animescore": 0.0, "cer": 1.0,
            "cer_violation": 1.0, "ok": 0.0}

    if not ids:
        print("\033[91mAS+CER-constraint: no tokens, R=0.0\033[0m")
        return zero

    try:
        resp = _remote_anime(ids)
        animescore = float(resp["raw_score"])
    except Exception as e:
        warnings.warn(f"Animescore server error: {e}; R=0")
        print("\033[91mAS+CER-constraint: anime ERROR, R=0.0\033[0m")
        return zero

    try:
        resp_w = _remote_whisper(ids, ground_truth)
        transcript = resp_w.get("transcript", "")
        c = float(cer(ground_truth, transcript)) if transcript else 1.0
    except Exception as e:
        warnings.warn(f"Whisper server error: {e}; assuming CER=1 (violation)")
        c = 1.0

    as_pos = max(0.0, animescore + animescore_offset)

    if c > cer_high:
        # Constraint violated — hard penalty, animescore gradient suppressed
        reward = violation_penalty
        violation = 1.0
        zone = "VIOLATE"
        color = "\033[91m"  # red
    elif c <= cer_low:
        reward = as_pos + low_cer_bonus
        violation = 0.0
        zone = "CLEAN"
        color = "\033[92m"  # green
    else:
        reward = as_pos
        violation = 0.0
        zone = "FEASIBLE"
        color = "\033[93m"  # yellow

    print(
        f"{color}AS={animescore:+.3f} (pos={as_pos:.3f}) CER={c:.3f} "
        f"[{zone}] -> R={reward:+.4f}\033[0m"
    )
    return {
        "score":         reward,
        "animescore":    animescore,
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
