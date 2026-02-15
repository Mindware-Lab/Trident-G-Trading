from datetime import UTC, datetime

from trident_trader.execution.sim_fill import Fill
from trident_trader.portfolio.book import PortfolioBook


def test_portfolio_realized_pnl_on_round_trip() -> None:
    b = PortfolioBook(initial_cash=10000)
    ts = datetime(2025, 1, 1, tzinfo=UTC)

    b.apply_fill(Fill("MES", "buy", 1, 100.0, 100.0, 0.0, 1.0, 0.5), ts)
    b.mark_to_market({"MES": 102.0}, ts)
    b.apply_fill(Fill("MES", "sell", 1, 102.0, 102.0, 0.0, 1.0, 0.5), ts)
    b.mark_to_market({"MES": 102.0}, ts)

    assert b.realized_pnl == 2.0
    assert b.qty("MES") == 0.0
