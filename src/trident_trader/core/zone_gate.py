from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneInputs:
    lambda_world: float
    load: float
    mismatch: float


def select_zone(inputs: ZoneInputs) -> str:
    if inputs.mismatch > 1.5 or inputs.load > 1.5:
        return "reset"
    if inputs.lambda_world < 0.4:
        return "light"
    return "full"
