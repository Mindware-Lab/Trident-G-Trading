from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trident_trader.core.type2_gate import hill_tail_exponent, interevent_times


@dataclass(frozen=True)
class UpdateTailDiagnostics:
    tail_exponent: float
    interevent_mean: float
    interevent_count: int


def update_tail_diagnostics(
    update_magnitudes: Sequence[float], big_update_steps: Sequence[int]
) -> UpdateTailDiagnostics:
    interevents = interevent_times(big_update_steps)
    interevent_mean = sum(interevents) / len(interevents) if interevents else 0.0
    return UpdateTailDiagnostics(
        tail_exponent=hill_tail_exponent(update_magnitudes),
        interevent_mean=interevent_mean,
        interevent_count=len(interevents),
    )
