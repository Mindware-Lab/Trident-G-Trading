from __future__ import annotations

from dataclasses import dataclass

from trident_trader.execution.sim_fill import Fill, simulate_fill
from trident_trader.world.schemas import Bar


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    qty: float


class SimulatedOMS:
    def __init__(self, slippage_bps: float = 0.5, fee_bps: float = 0.2) -> None:
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps

    def execute(self, intent: OrderIntent, bar: Bar) -> Fill:
        if intent.symbol != bar.symbol:
            raise ValueError("intent symbol and bar symbol must match")
        return simulate_fill(
            bar=bar,
            side=intent.side,
            qty=intent.qty,
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
        )
