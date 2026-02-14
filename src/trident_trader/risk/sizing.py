from __future__ import annotations


def type1_risk_multiplier(zone: str, lambda_world: float, mismatch_abs: float) -> float:
    zone_base = {
        "reset": 0.0,
        "light": 0.4,
        "full": 1.0,
    }.get(zone, 0.2)

    lambda_scale = max(0.0, min(1.0, lambda_world))
    mismatch_penalty = max(0.0, 1.0 - min(1.0, mismatch_abs / 0.5))
    return zone_base * (0.3 + 0.7 * lambda_scale) * mismatch_penalty
