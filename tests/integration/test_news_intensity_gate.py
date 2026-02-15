from __future__ import annotations

import heapq
from datetime import UTC, datetime, timedelta

from trident_trader.backtest.engine import BacktestEngine
from trident_trader.world.schemas import Bar, NewsEvent


def _bars(symbol: str) -> list[Bar]:
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    out: list[Bar] = []
    close = 100.0
    for i in range(120):
        ts = start + timedelta(minutes=i + 1)
        close *= 1.00005
        out.append(
            Bar(
                ts=ts,
                symbol=symbol,
                open=close,
                high=close * 1.0002,
                low=close * 0.9998,
                close=close,
                volume=1200.0,
                bid=close - 0.005,
                ask=close + 0.005,
            )
        )
    return out


def test_news_intensity_penalty_can_flip_gate() -> None:
    symbols = ["A", "B"]
    periods = {
        "fast": timedelta(minutes=5),
        "medium": timedelta(minutes=60),
        "slow": timedelta(days=1),
    }
    lambda_cfg = {
        "lambda": {
            "k_of_n": 2,
            "min_lambda_stream": 0.30,
            "min_lambda_global": 0.30,
            "lambda_clock": "medium",
            "regime_clock": "slow",
            "weights": {
                "liquidity": 0.25,
                "integrity": 0.15,
                "stability": 0.10,
                "event_penalty": 0.50,
            },
            "liquidity": {
                "use_bid_ask_if_available": True,
                "max_spread_bps": 2.5,
                "min_volume_z": -0.8,
            },
            "integrity": {
                "max_gap_rate": 0.01,
                "max_outlier_rate": 0.02,
            },
            "stability": {
                "max_vol_of_vol": 2.0,
                "max_corr_shock": 0.50,
            },
            "event_penalty": {
                "enabled": True,
                "max_event_intensity_z": 0.1,
            },
        }
    }

    decisions: list[dict[str, object]] = []
    engine = BacktestEngine(
        symbols=symbols,
        periods=periods,
        lambda_cfg=lambda_cfg,
        on_decision=decisions.append,
    )

    stream_a = _bars("A")
    stream_b = _bars("B")
    news = [
        NewsEvent(ts=datetime(2025, 1, 6, 1, 10, tzinfo=UTC), source="gdelt", intensity=500.0),
        NewsEvent(ts=datetime(2025, 1, 6, 1, 20, tzinfo=UTC), source="gdelt", intensity=500.0),
    ]
    events = list(heapq.merge(stream_a, stream_b, news, key=lambda e: e.ts))
    engine.run(events)

    assert len(decisions) >= 2
    first_gate = decisions[0]["gate"]
    second_gate = decisions[1]["gate"]
    assert first_gate["armed"] is True
    assert second_gate["inputs"]["A"]["event_intensity_z"] > 0.0
    assert second_gate["armed"] is False
