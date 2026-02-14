from datetime import UTC, datetime

from trident_trader.features.lambda_world import compute_lambda
from trident_trader.world.schemas import WorldSnapshot


def test_compute_lambda_bounded() -> None:
    snapshot = WorldSnapshot(
        ts=datetime.now(tz=UTC),
        spread_bps=1.0,
        depth_score=0.8,
        realized_vol=0.9,
        vol_of_vol=0.3,
        slippage_bps=0.7,
        latency_ms=20.0,
        fee_bps=0.4,
        news_intensity=5.0,
        news_tone=0.0,
    )
    result = compute_lambda(snapshot)
    assert 0.0 <= result.lambda_world <= 1.0