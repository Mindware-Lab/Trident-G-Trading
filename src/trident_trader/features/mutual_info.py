from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MISummary:
    value: float
    mean: float
    std: float
    falling: bool


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_std(values: Sequence[float], mean: float) -> float:
    if not values:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _discretize(values: Sequence[float], bins: int) -> list[int]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0 for _ in values]
    width = (hi - lo) / bins
    out: list[int] = []
    for value in values:
        idx = int((value - lo) / width)
        out.append(min(bins - 1, max(0, idx)))
    return out


def categorical_mutual_information(x: Sequence[str], y: Sequence[str]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    if n == 0:
        return 0.0

    px = Counter(x)
    py = Counter(y)
    pxy = Counter(zip(x, y, strict=True))
    mi = 0.0
    for (xi, yi), cxy in pxy.items():
        p_xy = cxy / n
        p_x = px[xi] / n
        p_y = py[yi] / n
        if p_xy > 0.0 and p_x > 0.0 and p_y > 0.0:
            mi += p_xy * math.log(p_xy / (p_x * p_y) + 1e-12)
    return max(0.0, mi)


def continuous_mutual_information(x: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    xd = _discretize(x, bins=bins)
    yd = _discretize(y, bins=bins)
    return categorical_mutual_information([str(v) for v in xd], [str(v) for v in yd])


def summarize_mi(history: Sequence[float], lookback: int = 20) -> MISummary:
    if not history:
        return MISummary(value=0.0, mean=0.0, std=0.0, falling=False)

    window = list(history[-lookback:])
    value = window[-1]
    mean = _safe_mean(window)
    std = _safe_std(window, mean)
    prev_mean = _safe_mean(window[:-1]) if len(window) > 1 else value
    falling = value < prev_mean - 0.25 * max(std, 1e-6)
    return MISummary(value=value, mean=mean, std=std, falling=falling)
