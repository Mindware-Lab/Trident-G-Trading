from datetime import UTC, datetime, timedelta

from trident_trader.world.consolidators import TimeBarConsolidator
from trident_trader.world.schemas import Bar


def _bar(ts: datetime, close: float) -> Bar:
    return Bar(
        ts=ts,
        symbol="MES",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_time_consolidator_emits_anchored_bars() -> None:
    emitted: list[Bar] = []
    cons = TimeBarConsolidator(symbol="MES", period=timedelta(minutes=5), on_bar=emitted.append)

    start = datetime(2025, 1, 1, 0, 1, tzinfo=UTC)
    for i in range(6):
        cons.update(_bar(start + timedelta(minutes=i), 100.0 + i))
    cons.flush()

    assert len(emitted) >= 2
    assert emitted[0].ts.minute % 5 == 0
    assert emitted[1].ts.minute % 5 == 0
