from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WorldSnapshot:
    """Canonical world-state input for Lambda scoring."""

    ts: datetime
    spread_bps: float
    depth_score: float
    realized_vol: float
    vol_of_vol: float
    slippage_bps: float
    latency_ms: float
    fee_bps: float
    news_intensity: float
    news_tone: float = 0.0


@dataclass(frozen=True)
class Bar:
    """Canonical OHLCV bar ending at ts (UTC)."""

    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class NewsEvent:
    ts: datetime
    source: str
    intensity: float
