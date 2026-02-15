from trident_trader.features.mutual_info import (
    categorical_mutual_information,
    continuous_mutual_information,
    estimate_mi_regression,
    summarize_mi,
)


def test_categorical_mi_independent_is_low() -> None:
    x = ["a", "a", "b", "b"]
    y = ["x", "y", "x", "y"]
    assert categorical_mutual_information(x, y) < 1e-3


def test_continuous_mi_dependent_is_positive() -> None:
    x = [0.1, 0.2, 0.3, 0.4, 0.5]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert continuous_mutual_information(x, y, bins=3) > 0.0


def test_summarize_mi_falling() -> None:
    summary = summarize_mi([0.2, 0.21, 0.19, 0.10, 0.08])
    assert summary.falling


def test_estimate_mi_regression_dependent_positive() -> None:
    features = [[float(i)] for i in range(1, 80)]
    targets = [2.0 * f[0] + 1.0 for f in features]
    result = estimate_mi_regression(features, targets)
    assert result.value > 0.0
