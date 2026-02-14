from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeFeatures:
    realized_vol: float
    vol_of_vol: float
    corr_spike: float


@dataclass(frozen=True)
class RegimeState:
    label: str
    confidence: float


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def classify_regime(features: RegimeFeatures) -> RegimeState:
    stress = (
        0.45 * _clamp01(features.realized_vol / 2.0)
        + 0.35 * _clamp01(features.vol_of_vol / 1.2)
        + 0.20 * _clamp01(features.corr_spike)
    )
    if stress >= 0.75:
        return RegimeState(label="shock", confidence=stress)
    if stress >= 0.45:
        return RegimeState(label="volatile", confidence=stress)
    return RegimeState(label="calm", confidence=1.0 - stress)
