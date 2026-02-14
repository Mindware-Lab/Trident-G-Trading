from datetime import UTC, datetime, timedelta

from trident_trader.backtest.engine import BacktestEngine
from trident_trader.world.schemas import Bar


def test_engine_runs_and_emits_decisions() -> None:
    symbols = ["MES", "ZN", "CL", "6E"]
    periods = {
        "fast": timedelta(minutes=5),
        "medium": timedelta(minutes=60),
        "slow": timedelta(days=1),
    }

    lambda_cfg = {
        "lambda": {
            "k_of_n": 3,
            "min_lambda_stream": 0.62,
            "min_lambda_global": 0.65,
            "weights": {
                "liquidity": 0.4,
                "integrity": 0.25,
                "stability": 0.25,
                "event_penalty": 0.1,
            },
            "liquidity": {
                "max_spread_bps": 2.5,
                "min_volume_z": -0.8,
            },
            "integrity": {
                "max_gap_rate": 0.002,
                "max_outlier_rate": 0.001,
            },
            "stability": {
                "max_vol_of_vol": 2.0,
                "max_corr_shock": 0.35,
            },
            "event_penalty": {
                "enabled": False,
                "max_event_intensity_z": 1.5,
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

    bars: list[Bar] = []
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    for i in range(120):
        ts = start + timedelta(minutes=i + 1)
        for s in symbols:
            bars.append(
                Bar(
                    ts=ts,
                    symbol=s,
                    open=100.0,
                    high=101.0,
                    low=99.5,
                    close=100.2,
                    volume=1000.0,
                    bid=100.19,
                    ask=100.21,
                )
            )

    engine.run(bars)

    assert decisions
    gate = decisions[0]["gate"]
    assert isinstance(gate["armed"], bool)
    assert set(gate["per_stream"]).issuperset(set(symbols))
