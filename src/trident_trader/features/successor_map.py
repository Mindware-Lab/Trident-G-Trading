from __future__ import annotations

import math
from dataclasses import dataclass

from trident_trader.features.relational_graph import RelationalState


@dataclass(frozen=True)
class SuccessorSnapshot:
    state_id: int
    uncertainty: float
    top_successors: tuple[tuple[int, float], ...]
    transition_entropy: float
    td_error_norm: float
    learned: bool


class SuccessorMap:
    def __init__(self, gamma: float = 0.95, alpha: float = 0.1) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self._state_to_id: dict[str, int] = {}
        self._id_to_state: list[str] = []
        self._m: list[list[float]] = []
        self._transition_counts: list[list[int]] = []
        self._prev_state: int | None = None

    def _ensure_state(self, key: str) -> int:
        if key in self._state_to_id:
            return self._state_to_id[key]
        idx = len(self._id_to_state)
        self._state_to_id[key] = idx
        self._id_to_state.append(key)

        for row in self._m:
            row.append(0.0)
        self._m.append([0.0 for _ in range(idx + 1)])

        for row_i in self._transition_counts:
            row_i.append(0)
        self._transition_counts.append([0 for _ in range(idx + 1)])
        return idx

    def infer_state(self, state: RelationalState) -> int:
        return self._ensure_state(state.motif_key)

    def update(self, state: RelationalState, learn: bool = True) -> SuccessorSnapshot:
        s = self.infer_state(state)
        td_norm = 0.0
        if self._prev_state is not None:
            prev = self._prev_state
            self._transition_counts[prev][s] += 1

            if learn:
                target = [0.0 for _ in range(len(self._m))]
                target[s] = 1.0
                td_err_sq = 0.0
                for j in range(len(self._m)):
                    td_target = target[j] + self.gamma * self._m[s][j]
                    td_error = td_target - self._m[prev][j]
                    self._m[prev][j] += self.alpha * td_error
                    td_err_sq += td_error * td_error
                td_norm = math.sqrt(td_err_sq)

        self._prev_state = s
        return self.snapshot(s, learned=learn, td_error_norm=td_norm)

    def _transition_entropy(self, state_id: int) -> float:
        counts = self._transition_counts[state_id]
        total = sum(counts)
        if total <= 0:
            return 1.0
        probs = [c / total for c in counts if c > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(probs))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def snapshot(
        self, state_id: int, top_k: int = 3, learned: bool = True, td_error_norm: float = 0.0
    ) -> SuccessorSnapshot:
        counts = self._transition_counts[state_id]
        total = sum(counts)
        if total <= 0:
            return SuccessorSnapshot(
                state_id=state_id,
                uncertainty=1.0,
                top_successors=(),
                transition_entropy=1.0,
                td_error_norm=td_error_norm,
                learned=learned,
            )

        probs = [c / total for c in counts]
        ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:top_k]
        transition_entropy = self._transition_entropy(state_id)
        uncertainty = min(1.0, max(0.0, 0.7 * transition_entropy + 0.3 * min(1.0, td_error_norm)))

        return SuccessorSnapshot(
            state_id=state_id,
            uncertainty=uncertainty,
            top_successors=tuple((idx, prob) for idx, prob in ranked if prob > 0),
            transition_entropy=transition_entropy,
            td_error_norm=td_error_norm,
            learned=learned,
        )
