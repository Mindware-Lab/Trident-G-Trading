from trident_trader.core.entropy_mi_controller import ControllerInputs, EntropyMIController


def test_controller_explore_when_mismatch_high() -> None:
    c = EntropyMIController()
    out = c.step(
        ControllerInputs(
            policy_scores=[0.5, 0.4],
            signal_mi=0.10,
            operator_mi=0.08,
            mismatch_value=0.35,
            lambda_world=0.7,
        )
    )
    assert out.mode == "explore"


def test_controller_exploit_when_stable() -> None:
    c = EntropyMIController()
    out = c.step(
        ControllerInputs(
            policy_scores=[0.6, 0.4],
            signal_mi=0.20,
            operator_mi=0.18,
            mismatch_value=0.02,
            lambda_world=0.8,
        )
    )
    assert out.mode == "exploit"
