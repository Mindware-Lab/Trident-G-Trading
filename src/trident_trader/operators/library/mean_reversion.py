from __future__ import annotations

from trident_trader.operators.base import Signal


class MeanReversionOperator:
    name = "mean_reversion"

    def on_event(self, event: dict[str, object]) -> Signal | None:
        _ = event
        return None
