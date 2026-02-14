from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from trident_trader.features.lambda_world import (
    LambdaConfig,
    LambdaInputs,
    lambda_score,
    spread_bps_from_bar,
)
from trident_trader.world.consolidators import TimeBarConsolidator
from trident_trader.world.schemas import Bar


@dataclass
class StreamState:
    fast: TimeBarConsolidator
    medium: TimeBarConsolidator
    slow: TimeBarConsolidator


class BacktestEngine:
    def __init__(
        self,
        symbols: list[str],
        periods: dict[str, timedelta],
        lambda_cfg: dict[str, Any],
        on_decision: Callable[[dict[str, object]], None],
    ) -> None:
        self.symbols = symbols
        self.periods = periods
        self.lambda_cfg = lambda_cfg
        self.on_decision = on_decision

        self.streams: dict[str, StreamState] = {}
        self.latest_medium: dict[str, Bar] = {}
        self.latest_slow: dict[str, Bar] = {}

        for symbol in symbols:
            self.streams[symbol] = StreamState(
                fast=TimeBarConsolidator(symbol=symbol, period=periods["fast"]),
                medium=TimeBarConsolidator(
                    symbol=symbol,
                    period=periods["medium"],
                    on_bar=self._make_medium_callback(symbol),
                ),
                slow=TimeBarConsolidator(
                    symbol=symbol,
                    period=periods["slow"],
                    on_bar=self._make_slow_callback(symbol),
                ),
            )

    def _make_medium_callback(self, symbol: str) -> Callable[[Bar], None]:
        def _callback(bar: Bar) -> None:
            self._on_medium(symbol, bar)

        return _callback

    def _make_slow_callback(self, symbol: str) -> Callable[[Bar], None]:
        def _callback(bar: Bar) -> None:
            self._on_slow(symbol, bar)

        return _callback

    def run(self, bars: Iterable[Bar]) -> None:
        for bar in bars:
            if bar.symbol not in self.streams:
                continue
            stream = self.streams[bar.symbol]
            stream.fast.update(bar)
            stream.medium.update(bar)
            stream.slow.update(bar)

        for stream in self.streams.values():
            stream.fast.flush()
            stream.medium.flush()
            stream.slow.flush()

    def _on_medium(self, symbol: str, bar: Bar) -> None:
        self.latest_medium[symbol] = bar
        if len(self.latest_medium) != len(self.symbols):
            return

        gate = self._lambda_gate()
        ctx: dict[str, object] = {
            "ts": bar.ts,
            "gate": gate,
            "medium_bars": dict(self.latest_medium),
            "slow_bars": dict(self.latest_slow),
        }
        self.on_decision(ctx)

    def _on_slow(self, symbol: str, bar: Bar) -> None:
        self.latest_slow[symbol] = bar

    def _lambda_gate(self) -> dict[str, object]:
        lambda_section = cast(LambdaConfig, self.lambda_cfg["lambda"])
        per_stream: dict[str, float] = {}

        for symbol, bar in self.latest_medium.items():
            inp = LambdaInputs(
                spread_bps=spread_bps_from_bar(bar),
                volume_z=0.0,
                gap_rate=0.0,
                outlier_rate=0.0,
                vol_of_vol=0.0,
                corr_shock=0.0,
                event_intensity_z=0.0,
            )
            per_stream[symbol] = lambda_score(inp, lambda_section)

        values = sorted(per_stream.values())
        median_idx = len(values) // 2
        lambda_global = values[median_idx] if values else 0.0

        good_streams = sum(1 for value in values if value >= lambda_section["min_lambda_stream"])
        armed = (
            good_streams >= lambda_section["k_of_n"]
            and lambda_global >= lambda_section["min_lambda_global"]
        )

        return {
            "per_stream": per_stream,
            "lambda_global": lambda_global,
            "good_streams": good_streams,
            "armed": armed,
        }
