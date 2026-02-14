from __future__ import annotations

import csv
import heapq
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from trident_trader.world.schemas import Bar


def iter_csv_bars(path: Path, symbol: str) -> Iterator[Bar]:
    """
    CSV columns expected:
    ts,open,high,low,close,volume,bid,ask

    ts is interpreted as bar END timestamp in UTC.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts = datetime.fromisoformat(row["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            else:
                ts = ts.astimezone(UTC)

            yield Bar(
                ts=ts,
                symbol=symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0) or 0.0),
                bid=float(row["bid"]) if row.get("bid") else None,
                ask=float(row["ask"]) if row.get("ask") else None,
            )


def merge_sorted(streams: Iterable[Iterable[Bar]]) -> Iterator[Bar]:
    return heapq.merge(*streams, key=lambda b: b.ts)
