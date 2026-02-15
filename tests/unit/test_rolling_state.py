from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trident_trader.features.rolling_state import RollingFeatureState


def test_rolling_state_gap_rate_tracks_missing_intervals() -> None:
    state = RollingFeatureState(expected_period=timedelta(minutes=1))
    t0 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    state.update(ts=t0, close=100.0, volume=1000.0)
    state.update(ts=t0 + timedelta(minutes=1), close=100.1, volume=1000.0)
    state.update(ts=t0 + timedelta(minutes=4), close=100.2, volume=900.0)

    metrics = state.metrics()
    assert metrics.gap_rate > 0.0


def test_rolling_state_outlier_rate_flags_spike() -> None:
    state = RollingFeatureState(expected_period=timedelta(minutes=1))
    t0 = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)

    close = 100.0
    for i in range(10):
        close *= 1.0001
        state.update(ts=t0 + timedelta(minutes=i), close=close, volume=1000.0)

    # Large jump to trigger outlier
    state.update(ts=t0 + timedelta(minutes=11), close=close * 1.08, volume=3000.0)
    metrics = state.metrics()

    assert metrics.outlier_rate > 0.0
