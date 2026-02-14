from trident_trader.features.lambda_world import LambdaInputs, lambda_score


def test_lambda_score_bounds() -> None:
    cfg = {
        "weights": {
            "liquidity": 0.4,
            "integrity": 0.25,
            "stability": 0.25,
            "event_penalty": 0.1,
        },
        "liquidity": {
            "max_spread_bps": 2.5,
            "min_volume_z": -0.8,
        },
        "integrity": {
            "max_gap_rate": 0.002,
            "max_outlier_rate": 0.001,
        },
        "stability": {
            "max_vol_of_vol": 2.0,
            "max_corr_shock": 0.35,
        },
        "event_penalty": {
            "enabled": True,
            "max_event_intensity_z": 1.5,
        },
    }
    score = lambda_score(
        LambdaInputs(
            spread_bps=1.2,
            volume_z=0.2,
            gap_rate=0.0,
            outlier_rate=0.0,
            vol_of_vol=0.4,
            corr_shock=0.1,
            event_intensity_z=0.6,
        ),
        cfg,
    )
    assert 0.0 <= score <= 1.0
