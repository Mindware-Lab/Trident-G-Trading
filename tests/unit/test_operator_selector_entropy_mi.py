from __future__ import annotations

from trident_trader.core.operator_selector_entropy_mi import (
    OperatorSelectorEntropyMI,
    SelectorConfig,
)


def test_selector_returns_flat_when_disarmed() -> None:
    selector = OperatorSelectorEntropyMI()
    op, mi = selector.select(armed=False, mismatch=0.0, feature_vector=[0.0, 0.0, 0.0])
    assert op == "flat"
    assert mi == 0.0


def test_selector_temperature_rises_with_high_mismatch() -> None:
    selector = OperatorSelectorEntropyMI(
        config=SelectorConfig(mi_n_min=999, tau_init=1.0, tau_step=0.2)
    )
    before = selector.temperature
    selector.select(armed=True, mismatch=0.5, feature_vector=[0.1, 0.2, 0.3])
    assert selector.temperature > before
