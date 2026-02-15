from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationalEdge:
    source: str
    target: str
    weight: float
    relation: str


@dataclass(frozen=True)
class RelationalState:
    top_edges: tuple[RelationalEdge, ...]
    coupling_index: float
    cluster_label: str
    motif_key: str
    vector: tuple[float, ...]


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sx = (sum((v - mx) ** 2 for v in x) / len(x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y) / len(y)) ** 0.5
    if sx <= 0 or sy <= 0:
        return 0.0
    cov = float(sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / len(x))
    denom = float(sx * sy)
    return float(cov / denom)


class RelationalGraphMap:
    def __init__(
        self,
        symbols: list[str],
        window: int = 120,
        edge_threshold: float = 0.25,
        ema_alpha: float = 0.2,
        hysteresis_steps: int = 2,
        top_k: int = 4,
    ) -> None:
        self.symbols = symbols
        self.window = window
        self.edge_threshold = edge_threshold
        self.ema_alpha = ema_alpha
        self.hysteresis_steps = hysteresis_steps
        self.top_k = top_k

        self._returns: dict[str, deque[float]] = {
            symbol: deque(maxlen=window) for symbol in symbols
        }
        self._edge_ema: dict[tuple[str, str], float] = defaultdict(float)
        self._edge_hits: dict[tuple[str, str], int] = defaultdict(int)

    def update(self, returns: dict[str, float]) -> RelationalState:
        for symbol in self.symbols:
            self._returns[symbol].append(float(returns.get(symbol, 0.0)))

        edges: list[RelationalEdge] = []
        abs_weights: list[float] = []
        for i, source in enumerate(self.symbols):
            for target in self.symbols[i + 1 :]:
                key = (source, target)
                corr = _corr(list(self._returns[source]), list(self._returns[target]))
                prev = self._edge_ema[key]
                ema = (1.0 - self.ema_alpha) * prev + self.ema_alpha * corr
                self._edge_ema[key] = ema
                abs_weights.append(abs(ema))

                if abs(ema) >= self.edge_threshold:
                    self._edge_hits[key] += 1
                else:
                    self._edge_hits[key] = 0

                relation = "none"
                if self._edge_hits[key] >= self.hysteresis_steps:
                    relation = "co_move_pos" if ema > 0 else "co_move_neg"
                edges.append(
                    RelationalEdge(source=source, target=target, weight=ema, relation=relation)
                )

        ranked = sorted(edges, key=lambda e: abs(e.weight), reverse=True)
        top_edges = tuple(ranked[: self.top_k])
        coupling = sum(abs_weights) / len(abs_weights) if abs_weights else 0.0
        cluster = "coupled" if coupling >= 0.5 else "fragmented"
        motif = "|".join(
            f"{e.source}>{e.target}:{'+' if e.weight >= 0 else '-'}" for e in top_edges[:3]
        )

        vector = tuple(
            [round(coupling, 6)]
            + [round(e.weight, 6) for e in top_edges[:3]]
            + [float(len(top_edges))]
        )
        return RelationalState(
            top_edges=top_edges,
            coupling_index=coupling,
            cluster_label=cluster,
            motif_key=motif,
            vector=vector,
        )
