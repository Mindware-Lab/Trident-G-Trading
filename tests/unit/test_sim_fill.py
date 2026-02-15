from datetime import UTC, datetime

from trident_trader.execution.sim_fill import simulate_fill
from trident_trader.world.schemas import Bar


def test_sim_fill_buy_uses_ask_and_worsens() -> None:
    bar = Bar(
        ts=datetime(2025, 1, 1, tzinfo=UTC),
        symbol="MES",
        open=100,
        high=101,
        low=99,
        close=100,
        volume=1,
        bid=99.99,
        ask=100.01,
    )
    fill = simulate_fill(bar=bar, side="buy", qty=1.0, slippage_bps=1.0)
    assert fill.price > 100.01
    assert fill.spread_bps > 0
