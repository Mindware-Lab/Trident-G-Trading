from __future__ import annotations

import math
from collections.abc import Sequence


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def softmax(scores: Sequence[float], temperature: float) -> list[float]:
    if not scores:
        return []
    tau = max(1e-6, temperature)
    scaled = [s / tau for s in scores]
    m = max(scaled)
    exp_vals = [math.exp(v - m) for v in scaled]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]


def shannon_entropy(probs: Sequence[float]) -> float:
    entropy = 0.0
    for p in probs:
        if p > 0.0:
            entropy -= p * math.log(p)
    return entropy


def normalized_entropy(probs: Sequence[float]) -> float:
    if not probs:
        return 0.0
    h = shannon_entropy(probs)
    max_h = math.log(len(probs))
    if max_h <= 0.0:
        return 0.0
    return h / max_h


def update_temperature(
    current_tau: float,
    entropy_value: float,
    entropy_target: float,
    step: float,
    tau_min: float,
    tau_max: float,
) -> float:
    error = entropy_target - entropy_value
    next_tau = current_tau + step * error
    return _clamp(next_tau, tau_min, tau_max)
