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


@dataclass(frozen=True)
class MIRegressionResult:
    value: float
    stable: bool
    samples: int
    method: str


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


def _pearson_abs(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = _safe_mean(x)
    my = _safe_mean(y)
    sx = _safe_std(x, mx)
    sy = _safe_std(y, my)
    if sx <= 0 or sy <= 0:
        return 0.0
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / len(x)
    return abs(cov / (sx * sy))


def estimate_mi_regression(
    features: Sequence[Sequence[float]], targets: Sequence[float]
) -> MIRegressionResult:
    if len(features) != len(targets):
        raise ValueError("features and targets length mismatch")
    n = len(targets)
    if n < 5:
        return MIRegressionResult(value=0.0, stable=False, samples=n, method="insufficient")

    try:
        import numpy as np
        from sklearn.feature_selection import mutual_info_regression  # type: ignore[import-untyped]

        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        mi = mutual_info_regression(x, y, random_state=7)
        value = float(np.mean(mi)) if len(mi) > 0 else 0.0
        return MIRegressionResult(value=max(0.0, value), stable=True, samples=n, method="sklearn")
    except Exception:
        # Fallback for environments without sklearn.
        cols = list(zip(*features, strict=True))
        corr_scores = [_pearson_abs(list(col), list(targets)) for col in cols]
        value = sum(corr_scores) / len(corr_scores) if corr_scores else 0.0
        return MIRegressionResult(value=value, stable=True, samples=n, method="fallback")


def rolling_mi_relevance(
    features_history: Sequence[Sequence[float]],
    rewards_history: Sequence[float],
    window: int = 240,
    n_min: int = 200,
) -> MIRegressionResult:
    n = min(len(features_history), len(rewards_history))
    if n == 0:
        return MIRegressionResult(value=0.0, stable=False, samples=0, method="empty")

    x = list(features_history[-window:]) if n > window else list(features_history)
    y = list(rewards_history[-window:]) if n > window else list(rewards_history)
    if len(x) < n_min:
        return MIRegressionResult(value=0.0, stable=False, samples=len(x), method="warmup")
    return estimate_mi_regression(x, y)
