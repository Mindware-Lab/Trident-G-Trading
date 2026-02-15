from __future__ import annotations

from trident_trader.features.relational_graph import RelationalEdge, RelationalState
from trident_trader.features.successor_map import SuccessorMap


def _state(key: str) -> RelationalState:
    return RelationalState(
        top_edges=(RelationalEdge(source="A", target="B", weight=0.5, relation="co_move_pos"),),
        coupling_index=0.5,
        cluster_label="coupled",
        motif_key=key,
        vector=(0.5, 0.5, 1.0),
    )


def test_successor_map_updates_transitions() -> None:
    sr = SuccessorMap(alpha=0.5, gamma=0.9)
    s1 = _state("a")
    s2 = _state("b")
    sr.update(s1)
    snap = sr.update(s2)
    assert snap.state_id in {0, 1}
    assert 0.0 <= snap.uncertainty <= 1.0


def test_sr_uncertainty_spikes_around_transition_change() -> None:
    sr = SuccessorMap(alpha=0.3, gamma=0.9)
    a = _state("a")
    b = _state("b")

    # Stable regime: a -> a repeatedly, low transition entropy.
    stable_unc: list[float] = []
    for _ in range(30):
        snap = sr.update(a, learn=True)
        stable_unc.append(snap.uncertainty)
    stable_level = stable_unc[-1]

    # Shock regime: alternate between a and b, uncertainty should rise.
    shock_unc: list[float] = []
    for i in range(20):
        snap = sr.update(a if i % 2 == 0 else b, learn=True)
        shock_unc.append(snap.uncertainty)

    assert max(shock_unc) > stable_level
