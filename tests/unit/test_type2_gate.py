from trident_trader.core.type2_gate import (
    Type2GateInputs,
    hill_tail_exponent,
    interevent_times,
    should_trigger_type2,
)


def test_type2_trigger_conditions() -> None:
    triggered = should_trigger_type2(
        Type2GateInputs(
            mismatch_history=[0.22, 0.25, 0.23, 0.21, 0.05, 0.24],
            mi_history=[0.20, 0.18, 0.16, 0.12],
            lambda_world=0.6,
        )
    )
    assert triggered


def test_interevent_times() -> None:
    assert interevent_times([2, 5, 11]) == [3, 6]


def test_tail_exponent_positive() -> None:
    alpha = hill_tail_exponent([0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.1], top_k=3)
    assert alpha > 0.0
