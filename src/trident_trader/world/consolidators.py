from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trident_trader.world.schemas import Bar


def floor_time(ts: datetime, period: timedelta) -> datetime:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)

    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = ts - epoch
    bucket = int(delta.total_seconds() // period.total_seconds())
    return epoch + (bucket * period)


@dataclass
class TimeBarConsolidator:
    symbol: str
    period: timedelta
    on_bar: Callable[[Bar], None] | None = None

    _bucket_start: datetime | None = None
    _open: float | None = None
    _high: float | None = None
    _low: float | None = None
    _close: float | None = None
    _volume: float = 0.0
    _last_bid: float | None = None
    _last_ask: float | None = None

    def update(self, bar: Bar) -> None:
        bucket_start = floor_time(bar.ts, self.period)

        if self._bucket_start is None:
            self._start_new_bucket(bucket_start, bar)
            return

        if bucket_start != self._bucket_start:
            self._emit(ts_end=self._bucket_start + self.period)
            self._start_new_bucket(bucket_start, bar)
            return

        self._close = bar.close
        self._high = max(self._high, bar.high) if self._high is not None else bar.high
        self._low = min(self._low, bar.low) if self._low is not None else bar.low
        self._volume += float(bar.volume)
        if bar.bid is not None:
            self._last_bid = bar.bid
        if bar.ask is not None:
            self._last_ask = bar.ask

    def flush(self) -> None:
        if self._bucket_start is None:
            return
        self._emit(ts_end=self._bucket_start + self.period)
        self._bucket_start = None

    def _start_new_bucket(self, bucket_start: datetime, bar: Bar) -> None:
        self._bucket_start = bucket_start
        self._open = bar.open
        self._high = bar.high
        self._low = bar.low
        self._close = bar.close
        self._volume = float(bar.volume)
        self._last_bid = bar.bid
        self._last_ask = bar.ask

    def _emit(self, ts_end: datetime) -> None:
        if (
            self._bucket_start is None
            or self._open is None
            or self._high is None
            or self._low is None
            or self._close is None
        ):
            return

        out = Bar(
            ts=ts_end,
            symbol=self.symbol,
            open=float(self._open),
            high=float(self._high),
            low=float(self._low),
            close=float(self._close),
            volume=float(self._volume),
            bid=self._last_bid,
            ask=self._last_ask,
        )
        if self.on_bar is not None:
            self.on_bar(out)
