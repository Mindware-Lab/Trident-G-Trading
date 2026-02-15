from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trident_trader.backtest.walkforward import (
    FoldWindow,
    build_walkforward_windows,
    config_fingerprint,
    split_events_for_window,
)
from trident_trader.world.schemas import Bar


def test_build_walkforward_windows() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 2, 1, tzinfo=UTC)
    windows = build_walkforward_windows(
        event_start=start,
        event_end=end,
        train_period=timedelta(days=10),
        test_period=timedelta(days=5),
        step_period=timedelta(days=5),
    )
    assert windows
    assert windows[0].train_start == start
    assert windows[0].test_start == start + timedelta(days=10)


def test_split_events_for_window() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    window = FoldWindow(
        fold_index=0,
        train_start=start,
        train_end=start + timedelta(days=2),
        test_start=start + timedelta(days=2),
        test_end=start + timedelta(days=3),
    )
    events = [
        Bar(
            ts=start + timedelta(hours=i * 12),
            symbol="A",
            open=1,
            high=1,
            low=1,
            close=1,
            volume=1,
        )
        for i in range(8)
    ]
    train, test = split_events_for_window(events, window)
    assert train
    assert test
    assert all(window.train_start <= e.ts < window.train_end for e in train)
    assert all(window.test_start <= e.ts < window.test_end for e in test)


def test_config_fingerprint_stable() -> None:
    cfg = {"a": 1, "b": {"x": 2}}
    assert config_fingerprint(cfg) == config_fingerprint(cfg)
