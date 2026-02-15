from __future__ import annotations

from trident_trader.features.relational_graph import RelationalGraphMap


def test_relational_graph_detects_coupling() -> None:
    graph = RelationalGraphMap(symbols=["A", "B"], edge_threshold_weak=0.1, hysteresis_steps=1)
    state = None
    for i in range(40):
        r = 0.001 * ((i % 5) - 2)
        state = graph.update({"A": r, "B": r})
    assert state is not None
    assert state.top_edges
    assert abs(state.top_edges[0].weight) > 0.1


def test_relational_state_key_switches_without_thrashing() -> None:
    graph = RelationalGraphMap(
        symbols=["A", "B"],
        decay=0.90,
        edge_threshold_weak=0.15,
        edge_threshold_strong=0.35,
        hysteresis_steps=2,
    )

    keys: list[str] = []
    # Regime 1: positive coupling.
    for i in range(80):
        r = 0.001 * ((i % 7) - 3)
        state = graph.update({"A": r, "B": r})
        keys.append(state.motif_key)

    # Regime 2: negative coupling.
    for i in range(80):
        r = 0.001 * ((i % 7) - 3)
        state = graph.update({"A": r, "B": -r})
        keys.append(state.motif_key)

    switches = sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1])
    assert switches >= 1
    assert switches <= 8
