from __future__ import annotations

from trident_trader.features.relational_graph import RelationalGraphMap


def test_relational_graph_detects_coupling() -> None:
    graph = RelationalGraphMap(symbols=["A", "B"], edge_threshold=0.1, hysteresis_steps=1)
    state = None
    for i in range(40):
        r = 0.001 * ((i % 5) - 2)
        state = graph.update({"A": r, "B": r})
    assert state is not None
    assert state.top_edges
    assert abs(state.top_edges[0].weight) > 0.1
    assert state.cluster_label in {"coupled", "fragmented"}
