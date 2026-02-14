from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Type2GateConfig:
    mismatch_threshold: float = 0.20
    persistent_window: int = 6
    min_persistent_points: int = 4
    min_lambda_world: float = 0.35
    min_mi_drop: float = 0.02
    big_update_threshold: float = 0.12


@dataclass(frozen=True)
class Type2GateInputs:
    mismatch_history: Sequence[float]
    mi_history: Sequence[float]
    lambda_world: float


def _is_mismatch_persistent(
    mismatch_history: Sequence[float],
    threshold: float,
    window: int,
    min_points: int,
) -> bool:
    if not mismatch_history:
        return False
    recent = mismatch_history[-window:]
    hits = sum(1 for v in recent if abs(v) >= threshold)
    return hits >= min_points


def _mi_is_falling(mi_history: Sequence[float], min_drop: float) -> bool:
    if len(mi_history) < 3:
        return False
    prev = sum(mi_history[-3:-1]) / 2.0
    current = mi_history[-1]
    return current <= prev - min_drop


def should_trigger_type2(inputs: Type2GateInputs, config: Type2GateConfig | None = None) -> bool:
    cfg = config or Type2GateConfig()
    mismatch_persistent = _is_mismatch_persistent(
        mismatch_history=inputs.mismatch_history,
        threshold=cfg.mismatch_threshold,
        window=cfg.persistent_window,
        min_points=cfg.min_persistent_points,
    )
    mi_falling = _mi_is_falling(inputs.mi_history, min_drop=cfg.min_mi_drop)
    learnable_world = inputs.lambda_world >= cfg.min_lambda_world
    return mismatch_persistent and mi_falling and learnable_world


def propose_heavy_tail_step(
    scale: float = 0.05, dof: float = 3.0, seed: int | None = None
) -> float:
    rng = random.Random(seed)
    z = rng.gauss(0.0, 1.0)
    chi_sq = rng.gammavariate(dof / 2.0, 2.0)
    t = z / math.sqrt(chi_sq / dof)
    return scale * t


def accept_type2_proposal(mi_gain: float, predictive_gain: float, risk_ok: bool) -> bool:
    return risk_ok and (mi_gain > 0.0 or predictive_gain > 0.0)


def interevent_times(event_steps: Sequence[int]) -> list[int]:
    if len(event_steps) < 2:
        return []
    out: list[int] = []
    for idx in range(1, len(event_steps)):
        out.append(event_steps[idx] - event_steps[idx - 1])
    return out


def hill_tail_exponent(samples: Sequence[float], top_k: int = 5) -> float:
    positive = sorted((abs(v) for v in samples if abs(v) > 0.0), reverse=True)
    if len(positive) <= top_k:
        return 0.0
    x_k = positive[top_k]
    if x_k <= 0.0:
        return 0.0
    s = sum(math.log(x / x_k) for x in positive[:top_k] if x > 0)
    if s <= 0.0:
        return 0.0
    return top_k / s
