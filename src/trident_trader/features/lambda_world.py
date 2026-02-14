from __future__ import annotations

from dataclasses import dataclass

from trident_trader.world.schemas import WorldSnapshot


@dataclass(frozen=True)
class LambdaComponents:
    liquidity: float
    regime: float
    friction: float
    news: float
    lambda_world: float


def _bounded_inverse(x: float, scale: float) -> float:
    x = max(0.0, x)
    return 1.0 / (1.0 + (x / scale))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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
