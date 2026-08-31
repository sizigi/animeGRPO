"""CER-zone gate shared by all CER-constrained reward axes.

Reward shape (paper eq. 1):
    R(c) = norm(p(c)) + bonus_clean        if CER(c) <= tau_low
    R(c) = norm(p(c))                       if tau_low < CER(c) <= tau_high
    R(c) = -rho                             if CER(c) > tau_high

Defaults match the paper: tau_low=0.10, tau_high=0.30, rho=1.0, bonus_clean=0.5.

`norm` is the axis-specific normalization passed by the caller (e.g.,
UTMOS / 5 for UTMOS, (Likability - 1) / 5 for Likability, AnimeScore + 3 / 6
for AnimeScore, arousal for VAD-Arousal).

This is a pure-Python utility with no model dependencies; it is called from
each tts_*_cer.py reward function after the axis predictor has produced a
raw score.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CerZoneConfig:
    tau_low:   float = 0.10
    tau_high:  float = 0.30
    rho:       float = 1.0
    bonus_clean: float = 0.5


def gated_reward(score_norm: float, cer: float | None,
                 cfg: CerZoneConfig = CerZoneConfig()) -> float:
    """Apply the CER-zone gate to a normalized axis score.

    Args:
        score_norm: axis-normalized score (caller is responsible for normalization).
                    Negative values are clipped to 0 to ensure that, in the
                    feasible zone, reward >= 0 (no negative reward unless violator).
        cer: Character Error Rate in [0, 1]. None = treated as violator.
        cfg: zone configuration.
    Returns:
        Gated reward (float).
    """
    if cer is None or cer > cfg.tau_high:
        return -cfg.rho
    base = max(0.0, float(score_norm))
    if cer <= cfg.tau_low:
        return base + cfg.bonus_clean
    return base
