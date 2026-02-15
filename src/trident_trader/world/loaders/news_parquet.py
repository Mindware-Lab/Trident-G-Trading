from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from trident_trader.world.schemas import NewsEvent


def _coerce_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        ts = raw
    else:
        ts = datetime.fromisoformat(str(raw))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def iter_news_events(
    path: Path,
    source: str = "gdelt",
    column_ts: str = "ts",
    column_intensity: str = "count",
) -> Iterator[NewsEvent]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pq.read_table(path, columns=[column_ts, column_intensity])
        ts_col = table[column_ts].to_pylist()
        intensity_col = table[column_intensity].to_pylist()
        for ts_raw, intensity_raw in zip(ts_col, intensity_col, strict=True):
            yield NewsEvent(ts=_coerce_ts(ts_raw), source=source, intensity=float(intensity_raw))
        return

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield NewsEvent(
                    ts=_coerce_ts(row[column_ts]),
                    source=source,
                    intensity=float(row[column_intensity]),
                )
        return

    raise ValueError(f"Unsupported news file extension: {path.suffix}")
