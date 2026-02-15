from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from trident_trader.features.lambda_world import (
    LambdaConfig,
    LambdaInputs,
    lambda_score,
    spread_bps_from_bar,
)
from trident_trader.features.rolling_state import RollingFeatureState
from trident_trader.world.consolidators import TimeBarConsolidator, floor_time
from trident_trader.world.schemas import Bar, NewsEvent


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
        self.rolling_state: dict[str, RollingFeatureState] = {}
        self.news_by_bucket_end: dict[datetime, float] = {}

        for symbol in symbols:
            self.rolling_state[symbol] = RollingFeatureState(expected_period=periods["medium"])
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

    def run(self, events: Iterable[Bar | NewsEvent]) -> None:
        for event in events:
            if isinstance(event, NewsEvent):
                self._on_news(event)
                continue

            bar = event
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
        event_intensity = self.news_by_bucket_end.get(bar.ts, 0.0)
        self.rolling_state[symbol].update(
            ts=bar.ts,
            close=bar.close,
            volume=bar.volume,
            event_intensity=event_intensity,
        )
        if len(self.latest_medium) != len(self.symbols):
            return

        # Prevent unbounded growth of old news buckets.
        stale_keys = [key for key in self.news_by_bucket_end if key < bar.ts]
        for key in stale_keys:
            self.news_by_bucket_end.pop(key, None)

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

    def _on_news(self, event: NewsEvent) -> None:
        bucket_start = floor_time(event.ts, self.periods["medium"])
        bucket_end = bucket_start + self.periods["medium"]
        prev = self.news_by_bucket_end.get(bucket_end, 0.0)
        self.news_by_bucket_end[bucket_end] = prev + event.intensity

    def _lambda_gate(self) -> dict[str, object]:
        lambda_section = cast(LambdaConfig, self.lambda_cfg["lambda"])
        per_stream: dict[str, float] = {}
        per_stream_inputs: dict[str, dict[str, float]] = {}

        returns = [self.rolling_state[symbol].metrics().last_return for symbol in self.symbols]
        ret_median = sorted(returns)[len(returns) // 2] if returns else 0.0

        for symbol, bar in self.latest_medium.items():
            m = self.rolling_state[symbol].metrics()
            corr_shock = abs(m.last_return - ret_median) * 100.0
            inp = LambdaInputs(
                spread_bps=spread_bps_from_bar(bar),
                volume_z=m.volume_z,
                gap_rate=m.gap_rate,
                outlier_rate=m.outlier_rate,
                vol_of_vol=m.vol_of_vol,
                corr_shock=corr_shock,
                event_intensity_z=m.event_intensity_z,
            )
            per_stream[symbol] = lambda_score(inp, lambda_section)
            per_stream_inputs[symbol] = {
                "volume_z": m.volume_z,
                "gap_rate": m.gap_rate,
                "outlier_rate": m.outlier_rate,
                "vol_of_vol": m.vol_of_vol,
                "corr_shock": corr_shock,
                "event_intensity_z": m.event_intensity_z,
                "last_return": m.last_return,
            }

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
            "inputs": per_stream_inputs,
            "lambda_global": lambda_global,
            "good_streams": good_streams,
            "armed": armed,
        }
