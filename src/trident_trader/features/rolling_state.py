from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        return data[mid]
    return 0.5 * (data[mid - 1] + data[mid])


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    med = _median(values)
    return _median([abs(x - med) for x in values])


@dataclass(frozen=True)
class RollingMetrics:
    volume_z: float
    gap_rate: float
    outlier_rate: float
    vol_of_vol: float
    event_intensity_z: float
    last_return: float


class RollingFeatureState:
    def __init__(
        self,
        expected_period: timedelta,
        volume_window: int = 60,
        return_window: int = 60,
        vol_window: int = 30,
        outlier_k_mad: float = 6.0,
        event_window: int = 60,
    ) -> None:
        self.expected_period = expected_period
        self.volume_window = volume_window
        self.return_window = return_window
        self.vol_window = vol_window
        self.outlier_k_mad = outlier_k_mad
        self.event_window = event_window

        self._last_ts: datetime | None = None
        self._last_close: float | None = None
        self._missing_intervals = 0
        self._total_intervals = 0

        self._volumes: deque[float] = deque(maxlen=volume_window)
        self._returns: deque[float] = deque(maxlen=return_window)
        self._vol_series: deque[float] = deque(maxlen=vol_window)
        self._outlier_flags: deque[int] = deque(maxlen=return_window)
        self._events: deque[float] = deque(maxlen=event_window)

        self._last_return = 0.0

    def update(
        self, ts: datetime, close: float, volume: float, event_intensity: float = 0.0
    ) -> None:
        if self._last_ts is not None:
            elapsed = ts - self._last_ts
            periods = int(elapsed.total_seconds() // self.expected_period.total_seconds())
            if periods > 0:
                self._total_intervals += periods
                self._missing_intervals += max(0, periods - 1)
            else:
                self._total_intervals += 1

        if self._last_close is not None and self._last_close > 0:
            ret = (close / self._last_close) - 1.0
        else:
            ret = 0.0
        self._last_return = ret
        self._returns.append(ret)
        self._volumes.append(volume)
        self._events.append(event_intensity)

        current_vol = _std(list(self._returns))
        self._vol_series.append(current_vol)

        ret_list = list(self._returns)
        med = _median(ret_list) if ret_list else 0.0
        mad = _mad(ret_list)
        threshold = self.outlier_k_mad * (mad if mad > 0 else 1e-9)
        is_outlier = 1 if abs(ret - med) > threshold and len(ret_list) > 5 else 0
        self._outlier_flags.append(is_outlier)

        self._last_ts = ts
        self._last_close = close

    def metrics(self) -> RollingMetrics:
        vol_list = list(self._volumes)
        mean_vol = sum(vol_list) / len(vol_list) if vol_list else 0.0
        std_vol = _std(vol_list)
        cur_vol = vol_list[-1] if vol_list else 0.0
        volume_z = (cur_vol - mean_vol) / std_vol if std_vol > 0 else 0.0

        gap_rate = (
            self._missing_intervals / self._total_intervals if self._total_intervals > 0 else 0.0
        )

        outlier_rate = (
            sum(self._outlier_flags) / len(self._outlier_flags) if self._outlier_flags else 0.0
        )

        vol_of_vol = _std(list(self._vol_series))

        ev = list(self._events)
        ev_mean = sum(ev) / len(ev) if ev else 0.0
        ev_std = _std(ev)
        ev_z = ((ev[-1] - ev_mean) / ev_std) if ev_std > 0 and ev else 0.0

        return RollingMetrics(
            volume_z=volume_z,
            gap_rate=gap_rate,
            outlier_rate=outlier_rate,
            vol_of_vol=vol_of_vol,
            event_intensity_z=ev_z,
            last_return=self._last_return,
        )
