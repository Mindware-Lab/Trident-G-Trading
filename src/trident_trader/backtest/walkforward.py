from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from trident_trader.world.schemas import Bar, NewsEvent


@dataclass(frozen=True)
class FoldWindow:
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def build_walkforward_windows(
    *,
    event_start: datetime,
    event_end: datetime,
    train_period: timedelta,
    test_period: timedelta,
    step_period: timedelta,
) -> list[FoldWindow]:
    windows: list[FoldWindow] = []
    cursor = event_start
    fold = 0
    while cursor + train_period + test_period <= event_end:
        train_start = cursor
        train_end = cursor + train_period
        test_start = train_end
        test_end = test_start + test_period
        windows.append(
            FoldWindow(
                fold_index=fold,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold += 1
        cursor += step_period
    return windows


def split_events_for_window(
    events: list[Bar | NewsEvent], window: FoldWindow
) -> tuple[list[Bar | NewsEvent], list[Bar | NewsEvent]]:
    train_events = [e for e in events if window.train_start <= e.ts < window.train_end]
    test_events = [e for e in events if window.test_start <= e.ts < window.test_end]
    return train_events, test_events


def config_fingerprint(config_bundle: dict[str, object]) -> str:
    encoded = json.dumps(config_bundle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold_window_to_dict(window: FoldWindow) -> dict[str, object]:
    raw = asdict(window)
    raw["train_start"] = window.train_start.isoformat()
    raw["train_end"] = window.train_end.isoformat()
    raw["test_start"] = window.test_start.isoformat()
    raw["test_end"] = window.test_end.isoformat()
    return raw
