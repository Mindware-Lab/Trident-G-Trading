from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from trident_trader.backtest.engine import BacktestEngine
from trident_trader.world.loaders.csv_bars import iter_csv_bars, merge_sorted


def _write_symbol_csv(path: Path, wide_spread_after: int | None = None) -> None:
    fieldnames = ["ts", "open", "high", "low", "close", "volume", "bid", "ask"]
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(120):
            ts = start + timedelta(minutes=idx + 1)
            close = 100.0 + 0.01 * idx
            spread_bps = 0.5
            if wide_spread_after is not None and idx >= wide_spread_after:
                spread_bps = 10.0
            spread_abs = close * spread_bps / 10000.0
            writer.writerow(
                {
                    "ts": ts.isoformat(),
                    "open": f"{close:.6f}",
                    "high": f"{(close + 0.02):.6f}",
                    "low": f"{(close - 0.02):.6f}",
                    "close": f"{close:.6f}",
                    "volume": "1000",
                    "bid": f"{(close - spread_abs / 2.0):.6f}",
                    "ask": f"{(close + spread_abs / 2.0):.6f}",
                }
            )


def test_csv_ingestion_ordering_and_gate_flip() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        path_a = root / "A.csv"
        path_b = root / "B.csv"
        _write_symbol_csv(path_a)
        _write_symbol_csv(path_b, wide_spread_after=60)

        stream_a = list(iter_csv_bars(path_a, symbol="A"))
        stream_b = list(iter_csv_bars(path_b, symbol="B"))
        merged = list(merge_sorted([stream_a, stream_b]))

        assert merged[0].ts <= merged[1].ts
        assert {merged[0].symbol, merged[1].symbol} == {"A", "B"}

        decisions: list[dict[str, object]] = []
        lambda_cfg = {
            "lambda": {
                "k_of_n": 2,
                "min_lambda_stream": 0.65,
                "min_lambda_global": 0.65,
                "lambda_clock": "medium",
                "regime_clock": "slow",
                "weights": {
                    "liquidity": 0.4,
                    "integrity": 0.25,
                    "stability": 0.25,
                    "event_penalty": 0.1,
                },
                "liquidity": {
                    "use_bid_ask_if_available": True,
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
        engine = BacktestEngine(
            symbols=["A", "B"],
            periods={
                "fast": timedelta(minutes=5),
                "medium": timedelta(minutes=60),
                "slow": timedelta(days=1),
            },
            lambda_cfg=lambda_cfg,
            on_decision=decisions.append,
        )
        engine.run(merged)

        assert len(decisions) >= 2
        assert decisions[0]["gate"]["armed"] is True
        assert decisions[-1]["gate"]["armed"] is False
        assert decisions[0]["ts"].minute == 0
        assert decisions[-1]["ts"].minute == 0
        assert "inputs" in decisions[-1]["gate"]
        assert "gap_rate" in decisions[-1]["gate"]["inputs"]["A"]
