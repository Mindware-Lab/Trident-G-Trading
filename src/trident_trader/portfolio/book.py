from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from trident_trader.execution.sim_fill import Fill


@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class EquityPoint:
    ts: datetime
    equity: float


class PortfolioBook:
    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.turnover = 0.0
        self._equity = initial_cash
        self._equity_curve: list[EquityPoint] = []
        self._daily_start_equity: dict[date, float] = {}
        self._daily_realized: dict[date, float] = {}

    def qty(self, symbol: str) -> float:
        return self.positions.get(symbol, Position()).qty

    def apply_fill(self, fill: Fill, ts: datetime) -> None:
        pos = self.positions.setdefault(fill.symbol, Position())
        signed_qty = fill.qty if fill.side == "buy" else -fill.qty

        self.cash -= signed_qty * fill.price
        self.cash -= fill.fee
        self.fees_paid += fill.fee
        self.turnover += fill.notional

        old_qty = pos.qty
        new_qty = old_qty + signed_qty

        if old_qty == 0 or old_qty * signed_qty > 0:
            # opening or adding same direction
            total_abs = abs(old_qty) + abs(signed_qty)
            if total_abs > 0:
                pos.avg_price = (
                    abs(old_qty) * pos.avg_price + abs(signed_qty) * fill.price
                ) / total_abs
            pos.qty = new_qty
        else:
            # reducing or flipping
            close_qty = min(abs(old_qty), abs(signed_qty))
            if old_qty > 0:
                pnl = close_qty * (fill.price - pos.avg_price)
            else:
                pnl = close_qty * (pos.avg_price - fill.price)
            self.realized_pnl += pnl
            d = ts.date()
            self._daily_realized[d] = self._daily_realized.get(d, 0.0) + pnl

            pos.qty = new_qty
            if pos.qty == 0:
                pos.avg_price = 0.0
            elif old_qty * new_qty < 0:
                # flipped direction, new entry at fill
                pos.avg_price = fill.price

    def mark_to_market(self, prices: dict[str, float], ts: datetime) -> float:
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            if pos.qty == 0:
                continue
            px = prices.get(symbol)
            if px is None:
                continue
            unrealized += pos.qty * (px - pos.avg_price)

        self._equity = self.cash + unrealized
        self._equity_curve.append(EquityPoint(ts=ts, equity=self._equity))

        d = ts.date()
        self._daily_start_equity.setdefault(d, self._equity)
        return self._equity

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def equity_curve(self) -> list[EquityPoint]:
        return list(self._equity_curve)

    def daily_pnl(self, ts: datetime) -> float:
        d = ts.date()
        start = self._daily_start_equity.get(d, self.initial_cash)
        return self._equity - start

    def daily_realized_pnl(self, ts: datetime) -> float:
        return self._daily_realized.get(ts.date(), 0.0)
