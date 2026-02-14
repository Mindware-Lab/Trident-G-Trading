from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from trident_trader.world.schemas import Bar, WorldSnapshot


class LambdaWeights(TypedDict):
    liquidity: float
    integrity: float
    stability: float
    event_penalty: float


class LambdaLiquidity(TypedDict):
    use_bid_ask_if_available: bool
    max_spread_bps: float
    min_volume_z: float


class LambdaIntegrity(TypedDict):
    max_gap_rate: float
    max_outlier_rate: float


class LambdaStability(TypedDict):
    max_vol_of_vol: float
    max_corr_shock: float


class LambdaEventPenalty(TypedDict):
    enabled: bool
    max_event_intensity_z: float


class LambdaConfig(TypedDict):
    k_of_n: int
    min_lambda_stream: float
    min_lambda_global: float
    lambda_clock: str
    regime_clock: str
    weights: LambdaWeights
    liquidity: LambdaLiquidity
    integrity: LambdaIntegrity
    stability: LambdaStability
    event_penalty: LambdaEventPenalty


@dataclass(frozen=True)
class LambdaComponents:
    liquidity: float
    regime: float
    friction: float
    news: float
    lambda_world: float


@dataclass(frozen=True)
class LambdaInputs:
    spread_bps: float | None
    volume_z: float
    gap_rate: float
    outlier_rate: float
    vol_of_vol: float
    corr_shock: float
    event_intensity_z: float


def _bounded_inverse(x: float, scale: float) -> float:
    x = max(0.0, x)
    return 1.0 / (1.0 + (x / scale))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_spread(spread_bps: float | None, max_spread_bps: float) -> float:
    if spread_bps is None:
        return 0.6
    return _clamp01(1.0 - (spread_bps / max_spread_bps))


def score_volume(volume_z: float, min_volume_z: float) -> float:
    scaled = (volume_z - min_volume_z) / (2.0 - min_volume_z)
    return _clamp01(scaled)


def score_rate(x: float, max_x: float) -> float:
    if max_x <= 0:
        return 0.0
    return _clamp01(1.0 - (x / max_x))


def lambda_score(inp: LambdaInputs, cfg: LambdaConfig) -> float:
    w = cfg["weights"]
    liq_cfg = cfg["liquidity"]
    integ_cfg = cfg["integrity"]
    stab_cfg = cfg["stability"]
    evt_cfg = cfg["event_penalty"]

    liquidity = 0.5 * score_spread(inp.spread_bps, liq_cfg["max_spread_bps"]) + 0.5 * score_volume(
        inp.volume_z, liq_cfg["min_volume_z"]
    )

    integrity = 0.5 * score_rate(inp.gap_rate, integ_cfg["max_gap_rate"]) + 0.5 * score_rate(
        inp.outlier_rate, integ_cfg["max_outlier_rate"]
    )

    stability = 0.5 * score_rate(inp.vol_of_vol, stab_cfg["max_vol_of_vol"]) + 0.5 * score_rate(
        inp.corr_shock, stab_cfg["max_corr_shock"]
    )

    penalty = 0.0
    if evt_cfg["enabled"]:
        penalty = _clamp01((inp.event_intensity_z - evt_cfg["max_event_intensity_z"]) / 3.0)

    raw = (w["liquidity"] * liquidity) + (w["integrity"] * integrity) + (w["stability"] * stability)
    raw -= w["event_penalty"] * penalty
    return _clamp01(raw)


def spread_bps_from_bar(bar: Bar) -> float | None:
    if bar.bid is None or bar.ask is None:
        return None
    mid = 0.5 * (bar.bid + bar.ask)
    if mid <= 0:
        return None
    return 10000.0 * (bar.ask - bar.bid) / mid


def compute_lambda(snapshot: WorldSnapshot) -> LambdaComponents:
    """Compute Lambda(t) as world viability score in [0,1]."""
    spread_score = _bounded_inverse(snapshot.spread_bps, scale=8.0)
    depth_score = _clamp01(snapshot.depth_score)
    liquidity = 0.55 * spread_score + 0.45 * depth_score

    vol_score = _bounded_inverse(snapshot.realized_vol, scale=1.2)
    vov_score = _bounded_inverse(snapshot.vol_of_vol, scale=0.7)
    regime = 0.6 * vol_score + 0.4 * vov_score

    slippage_score = _bounded_inverse(snapshot.slippage_bps, scale=4.0)
    latency_score = _bounded_inverse(snapshot.latency_ms, scale=60.0)
    fee_score = _bounded_inverse(snapshot.fee_bps, scale=2.0)
    friction = 0.5 * slippage_score + 0.3 * latency_score + 0.2 * fee_score

    intensity_penalty = _bounded_inverse(snapshot.news_intensity, scale=30.0)
    tone_adjust = 1.0 - min(0.25, abs(snapshot.news_tone) / 100.0)
    news = _clamp01(intensity_penalty * tone_adjust)

    lambda_world = _clamp01(0.35 * liquidity + 0.25 * regime + 0.25 * friction + 0.15 * news)
    return LambdaComponents(
        liquidity=liquidity,
        regime=regime,
        friction=friction,
        news=news,
        lambda_world=lambda_world,
    )
