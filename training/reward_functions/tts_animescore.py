"""Raw reward for LLaSA-TTS using Animescore (voice quality) model.

The animescore model outputs an unbounded scalar (RankNet-trained).
GRPO already normalizes via (score - group_mean) / (group_std + eps),
so we return the raw score directly to preserve reward variance.

Requires a running animescore server (tts/animescore_server.py).
"""

from __future__ import annotations

import os
import re
import warnings
import time
from typing import List

import requests

# ---------------------------------------------------------------------------
# 1.  Token parsing (same as tts_cer.py)
# ---------------------------------------------------------------------------
def _parse_ids(token_str: str) -> List[int]:
    return [int(t) for t in re.findall(r"<\|s_(\d+)\|>", token_str)]

# ---------------------------------------------------------------------------
# 2.  Animescore server client helpers
# ---------------------------------------------------------------------------
SERVER = os.getenv("ANIMESCORE_SERVER", "http://localhost:8002")
SCORE_URL = f"{SERVER.rstrip('/')}/score"
HEALTH_URL = f"{SERVER.rstrip('/')}/healthz"

_last_health = 0.0


def _check_server():
    global _last_health
    if time.time() - _last_health < 30:
        return
    try:
        requests.get(HEALTH_URL, timeout=10)
        _last_health = time.time()
    except Exception as e:
        raise RuntimeError(f"Animescore server not reachable at {SERVER}: {e}")


def _remote_animescore(tokens: List[int]) -> dict:
    _check_server()
    payload = {"tokens": tokens}
    r = requests.post(SCORE_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()  # {"raw_score": float}

# ---------------------------------------------------------------------------
# 3.  Reward computation
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Return raw animescore as reward.

    GRPO normalizes via (score - group_mean) / (group_std + eps),
    so sigmoid is unnecessary and compresses variance.
    """
    ids = _parse_ids(solution_str)

    if not ids:
        print(f"\033[91mAnimescore: no tokens, Reward: 0.0000\033[0m")
        return 0.0

    try:
        resp = _remote_animescore(ids)
        raw_score = float(resp["raw_score"])
    except Exception as e:
        warnings.warn(f"Animescore server error: {e}; returning 0")
        print(f"\033[91mAnimescore: ERROR, Reward: 0.0000\033[0m")
        return 0.0

    print(f"\033[94mAnimescore: {raw_score:.4f}\033[0m")
    return raw_score


# CLI quick test
if __name__ == "__main__":
    import sys
    import json
    print(json.dumps(
        {"reward": compute_score("cli", sys.argv[1], sys.argv[2])},
        indent=2, ensure_ascii=False,
    ))
