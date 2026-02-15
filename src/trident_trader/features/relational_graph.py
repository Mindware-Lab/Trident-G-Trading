from __future__ import annotations

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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RelationalGraphMap:
    """
    Dynamic graph over stream returns.

    v0.1 implementation:
    - EWMA means/covariances
    - correlation edges
    - quantized motif with hysteresis to avoid thrashing
    """

    def __init__(
        self,
        symbols: list[str],
        decay: float = 0.94,
        edge_threshold_weak: float = 0.20,
        edge_threshold_strong: float = 0.50,
        hysteresis_steps: int = 2,
        top_k: int = 4,
    ) -> None:
        self.symbols = symbols
        self.decay = decay
        self.alpha = 1.0 - decay
        self.edge_threshold_weak = edge_threshold_weak
        self.edge_threshold_strong = edge_threshold_strong
        self.hysteresis_steps = hysteresis_steps
        self.top_k = top_k

        self._mean: dict[str, float] = {s: 0.0 for s in symbols}
        self._cov: dict[tuple[str, str], float] = {}
        self._stable_label: dict[tuple[str, str], str] = {}
        self._pending_label: dict[tuple[str, str], str] = {}
        self._pending_count: dict[tuple[str, str], int] = {}

        for i, source in enumerate(symbols):
            for target in symbols[i:]:
                self._cov[self._cov_key(source, target)] = 0.0
            for target in symbols[i + 1 :]:
                key = self._cov_key(source, target)
                self._stable_label[key] = "0"
                self._pending_label[key] = "0"
                self._pending_count[key] = 0

    def _cov_key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _quantize_label(self, corr: float) -> str:
        abs_corr = abs(corr)
        if abs_corr < self.edge_threshold_weak:
            return "0"
        if abs_corr < self.edge_threshold_strong:
            return "+1" if corr >= 0 else "-1"
        return "+2" if corr >= 0 else "-2"

    def update(self, returns: dict[str, float]) -> RelationalState:
        prev_mean = dict(self._mean)
        for symbol in self.symbols:
            r = float(returns.get(symbol, 0.0))
            self._mean[symbol] = self.decay * self._mean[symbol] + self.alpha * r

        # EWMA covariance with demeaned returns (using previous mean for stable update).
        for i, source in enumerate(self.symbols):
            x = float(returns.get(source, 0.0)) - prev_mean[source]
            for target in self.symbols[i:]:
                y = float(returns.get(target, 0.0)) - prev_mean[target]
                key = self._cov_key(source, target)
                self._cov[key] = self.decay * self._cov[key] + self.alpha * (x * y)

        edges: list[RelationalEdge] = []
        abs_weights: list[float] = []
        motif_parts: list[str] = []

        for i, source in enumerate(self.symbols):
            var_s = max(self._cov[(source, source)], 1e-12)
            for target in self.symbols[i + 1 :]:
                var_t = max(self._cov[(target, target)], 1e-12)
                key = self._cov_key(source, target)
                cov_st = self._cov[key]
                corr = _clamp(cov_st / ((var_s * var_t) ** 0.5), -1.0, 1.0)

                candidate = self._quantize_label(corr)
                if candidate == self._stable_label[key]:
                    self._pending_count[key] = 0
                else:
                    if candidate == self._pending_label[key]:
                        self._pending_count[key] += 1
                    else:
                        self._pending_label[key] = candidate
                        self._pending_count[key] = 1
                    if self._pending_count[key] >= self.hysteresis_steps:
                        self._stable_label[key] = candidate
                        self._pending_count[key] = 0

                relation = f"corr_{self._stable_label[key]}"
                edges.append(
                    RelationalEdge(source=source, target=target, weight=corr, relation=relation)
                )
                abs_weights.append(abs(corr))
                motif_parts.append(f"{source}>{target}:{self._stable_label[key]}")

        ranked = sorted(edges, key=lambda e: abs(e.weight), reverse=True)
        top_edges = tuple(ranked[: self.top_k])
        coupling = sum(abs_weights) / len(abs_weights) if abs_weights else 0.0

        strong_edges = sum(
            1
            for edge in edges
            if abs(edge.weight) >= self.edge_threshold_strong and edge.relation != "corr_0"
        )
        if strong_edges >= max(1, len(edges) // 2):
            cluster = "regime-coupled"
        elif coupling < self.edge_threshold_weak:
            cluster = "fragmented"
        else:
            cluster = "mixed"

        motif = "|".join(sorted(motif_parts))
        vector = tuple(
            [round(coupling, 6)]
            + [round(edge.weight, 6) for edge in top_edges[:3]]
            + [float(strong_edges)]
        )
        return RelationalState(
            top_edges=top_edges,
            coupling_index=coupling,
            cluster_label=cluster,
            motif_key=motif,
            vector=vector,
        )
