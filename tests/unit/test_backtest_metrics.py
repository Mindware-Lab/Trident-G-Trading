from datetime import UTC, datetime, timedelta

from trident_trader.backtest.metrics import summarize
from trident_trader.portfolio.book import EquityPoint


def test_summarize_backtest_stats() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    curve = [
        EquityPoint(ts=start, equity=100.0),
        EquityPoint(ts=start + timedelta(hours=1), equity=110.0),
        EquityPoint(ts=start + timedelta(hours=2), equity=105.0),
    ]
    stats = summarize(curve, turnover=1000.0, spread_samples=[1.0, 2.0], slippage_samples=[0.5])
    assert stats.total_return > 0
    assert stats.max_drawdown > 0
    assert stats.turnover == 1000.0
