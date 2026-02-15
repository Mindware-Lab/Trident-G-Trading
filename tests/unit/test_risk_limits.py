from trident_trader.risk.limits import RiskLimits, RiskState, check_order


def test_risk_limits_blocks_after_daily_loss() -> None:
    limits = RiskLimits(
        max_notional_per_order=1000.0, max_daily_loss=100.0, max_gross_notional=2000.0
    )
    state = RiskState()

    ok = check_order(
        limits=limits,
        state=state,
        order_notional=100.0,
        gross_notional_after=500.0,
        daily_pnl=-150.0,
        reduce_only=False,
    )
    assert not ok
    assert state.kill_switch
