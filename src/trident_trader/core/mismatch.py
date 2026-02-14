from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadInputs:
    realized_vol: float
    drawdown_velocity: float
    error_rate: float
    slippage_spike: float


@dataclass(frozen=True)
class BurdenInputs:
    forecast_vol: float
    expected_slippage: float
    event_risk: float


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_load(inputs: LoadInputs) -> float:
    return (
        0.35 * _clamp01(inputs.realized_vol / 2.0)
        + 0.30 * _clamp01(inputs.drawdown_velocity)
        + 0.20 * _clamp01(inputs.error_rate)
        + 0.15 * _clamp01(inputs.slippage_spike)
    )


def expected_burden(inputs: BurdenInputs) -> float:
    return (
        0.5 * _clamp01(inputs.forecast_vol / 2.0)
        + 0.3 * _clamp01(inputs.expected_slippage)
        + 0.2 * _clamp01(inputs.event_risk)
    )


def mismatch(load: float, expected_burden_value: float) -> float:
    return load - expected_burden_value
