from __future__ import annotations

from dataclasses import dataclass

from trident_trader.portfolio.book import EquityPoint


@dataclass(frozen=True)
class BacktestStats:
    total_return: float
    max_drawdown: float
    turnover: float
    avg_spread_bps: float
    avg_slippage_bps: float


def _max_drawdown(equity_curve: list[EquityPoint]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0].equity
    max_dd = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        if peak > 0:
            dd = (peak - point.equity) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def summarize(
    equity_curve: list[EquityPoint],
    turnover: float,
    spread_samples: list[float],
    slippage_samples: list[float],
) -> BacktestStats:
    if not equity_curve:
        return BacktestStats(0.0, 0.0, turnover, 0.0, 0.0)

    start = equity_curve[0].equity
    end = equity_curve[-1].equity
    total_return = (end - start) / start if start else 0.0
    avg_spread = sum(spread_samples) / len(spread_samples) if spread_samples else 0.0
    avg_slippage = sum(slippage_samples) / len(slippage_samples) if slippage_samples else 0.0

    return BacktestStats(
        total_return=total_return,
        max_drawdown=_max_drawdown(equity_curve),
        turnover=turnover,
        avg_spread_bps=avg_spread,
        avg_slippage_bps=avg_slippage,
    )
