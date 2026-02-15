from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_notional_per_order: float
    max_daily_loss: float
    max_gross_notional: float


@dataclass
class RiskState:
    kill_switch: bool = False


def check_order(
    limits: RiskLimits,
    state: RiskState,
    order_notional: float,
    gross_notional_after: float,
    daily_pnl: float,
    reduce_only: bool,
) -> bool:
    if state.kill_switch and not reduce_only:
        return False
    if daily_pnl <= -abs(limits.max_daily_loss):
        state.kill_switch = True
        if not reduce_only:
            return False
    if order_notional > limits.max_notional_per_order and not reduce_only:
        return False
    if gross_notional_after > limits.max_gross_notional and not reduce_only:
        return False
    return True
