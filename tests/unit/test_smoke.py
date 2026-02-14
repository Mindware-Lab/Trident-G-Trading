import pytest

from trident_trader.core.mismatch import mismatch
from trident_trader.core.zone_gate import ZoneInputs, select_zone


def test_mismatch() -> None:
    assert mismatch(1.2, 1.0) == pytest.approx(0.2)


def test_zone_gate_full() -> None:
    zone = select_zone(ZoneInputs(lambda_world=0.8, load=0.9, mismatch=0.1))
    assert zone == "full"