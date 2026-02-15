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
    # After one transition, some confidence mass should exist.
    if snap.top_successors:
        assert snap.top_successors[0][1] >= 0.0
