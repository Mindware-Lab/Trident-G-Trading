from __future__ import annotations

from dataclasses import dataclass

from trident_trader.world.schemas import Bar


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    notional: float
    fee: float
    spread_bps: float
    slippage_bps: float


def _infer_spread_bps(bar: Bar) -> float:
    if bar.bid is not None and bar.ask is not None and bar.close > 0:
        mid = 0.5 * (bar.bid + bar.ask)
        if mid > 0:
            return 10000.0 * (bar.ask - bar.bid) / mid
    # conservative fallback if no bid/ask
    return 2.0


def simulate_fill(
    bar: Bar,
    side: str,
    qty: float,
    slippage_bps: float = 0.5,
    fee_bps: float = 0.2,
) -> Fill:
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if qty <= 0:
        raise ValueError("qty must be > 0")

    spread_bps = _infer_spread_bps(bar)

    if bar.bid is not None and bar.ask is not None:
        base = bar.ask if side == "buy" else bar.bid
    else:
        # no quotes: cross half-spread around close
        half = bar.close * (spread_bps / 10000.0) / 2.0
        base = bar.close + half if side == "buy" else bar.close - half

    # worsen fill by slippage in trade direction
    slip_abs = base * (slippage_bps / 10000.0)
    price = base + slip_abs if side == "buy" else base - slip_abs
    notional = abs(price * qty)
    fee = notional * (fee_bps / 10000.0)

    return Fill(
        symbol=bar.symbol,
        side=side,
        qty=qty,
        price=price,
        notional=notional,
        fee=fee,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
    )
