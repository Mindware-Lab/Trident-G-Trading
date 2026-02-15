from __future__ import annotations

import heapq
import random
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from trident_trader.world.loaders.csv_bars import iter_csv_bars, merge_sorted
from trident_trader.world.loaders.news_parquet import iter_news_events
from trident_trader.world.schemas import Bar, NewsEvent


def parse_duration(value: str) -> timedelta:
    unit = value[-1]
    count = int(value[:-1])
    if unit == "m":
        return timedelta(minutes=count)
    if unit == "h":
        return timedelta(hours=count)
    if unit == "d":
        return timedelta(days=count)
    raise ValueError(f"Unsupported duration: {value}")


def load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def generate_smoke_bars(symbols: list[str], base_period: timedelta, steps: int = 400) -> list[Bar]:
    rng = random.Random(7)
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)
    bars: list[Bar] = []
    base_prices = {symbol: 100.0 + (idx * 20.0) for idx, symbol in enumerate(symbols)}

    for step in range(steps):
        ts = start + (step + 1) * base_period
        for symbol in symbols:
            drift = rng.uniform(-0.003, 0.003)
            vol = rng.uniform(0.001, 0.008)
            prev = base_prices[symbol]
            close = max(1.0, prev * (1.0 + drift))
            high = max(prev, close) * (1.0 + vol)
            low = min(prev, close) * (1.0 - vol)
            spread = close * rng.uniform(0.00002, 0.00008)
            bid = close - spread / 2.0
            ask = close + spread / 2.0
            volume = rng.uniform(200.0, 2000.0)
            bars.append(
                Bar(
                    ts=ts,
                    symbol=symbol,
                    open=prev,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    bid=bid,
                    ask=ask,
                )
            )
            base_prices[symbol] = close

    bars.sort(key=lambda b: b.ts)
    return bars


def load_news_events(
    *,
    data_root: Path,
    news_cfg: dict[str, object] | None,
    news_file_override: str | None,
) -> list[NewsEvent]:
    if news_file_override:
        path = data_root / news_file_override
        if not path.exists():
            raise FileNotFoundError(f"Missing news file override: {path}")
        return list(iter_news_events(path=path))

    if news_cfg is None:
        return []

    news = cast(dict[str, object], news_cfg.get("news", {}))
    out = cast(dict[str, object], news.get("output", {}))
    path_value = cast(str | None, out.get("path"))
    if not path_value:
        return []

    path = data_root / path_value
    if not path.exists():
        return []

    source = cast(str, news.get("source", "gdelt"))
    col_ts = cast(str, out.get("column_ts", "ts"))
    col_intensity = cast(str, out.get("column_intensity", "count"))
    return list(
        iter_news_events(
            path=path,
            source=source,
            column_ts=col_ts,
            column_intensity=col_intensity,
        )
    )


def load_events_from_universe(
    *,
    streams_cfg: list[dict[str, Any]],
    data_root: Path,
    news_cfg: dict[str, object] | None,
    news_file_override: str | None = None,
) -> list[Bar | NewsEvent]:
    bar_streams: list[list[Bar]] = []
    for stream in streams_cfg:
        symbol = cast(str, stream["symbol"])
        data_file = cast(str | None, stream.get("data_file"))
        if not data_file:
            raise ValueError(f"Universe stream for {symbol} is missing required data_file field.")
        path = data_root / data_file
        if not path.exists():
            raise FileNotFoundError(f"Missing data file for {symbol}: {path}")
        bar_streams.append(list(iter_csv_bars(path=path, symbol=symbol)))

    bars = list(merge_sorted(bar_streams))
    news_events = load_news_events(
        data_root=data_root,
        news_cfg=news_cfg,
        news_file_override=news_file_override,
    )
    merged = heapq.merge(
        cast(list[Bar | NewsEvent], bars),
        cast(list[Bar | NewsEvent], news_events),
        key=lambda e: e.ts,
    )
    return list(merged)
