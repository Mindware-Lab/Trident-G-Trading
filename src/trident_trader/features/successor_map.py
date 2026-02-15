from __future__ import annotations

from dataclasses import dataclass

from trident_trader.features.relational_graph import RelationalState


@dataclass(frozen=True)
class SuccessorSnapshot:
    state_id: int
    uncertainty: float
    top_successors: tuple[tuple[int, float], ...]


class SuccessorMap:
    def __init__(self, gamma: float = 0.95, alpha: float = 0.1) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self._state_to_id: dict[str, int] = {}
        self._id_to_state: list[str] = []
        self._m: list[list[float]] = []
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
        return idx

    def infer_state(self, state: RelationalState) -> int:
        return self._ensure_state(state.motif_key)

    def update(self, state: RelationalState) -> SuccessorSnapshot:
        s = self.infer_state(state)
        if self._prev_state is not None:
            prev = self._prev_state
            target = [0.0 for _ in range(len(self._m))]
            target[s] = 1.0
            for j in range(len(self._m)):
                boot = self.gamma * self._m[s][j]
                td_target = target[j] + boot
                self._m[prev][j] += self.alpha * (td_target - self._m[prev][j])
        self._prev_state = s
        return self.snapshot(s)

    def snapshot(self, state_id: int, top_k: int = 3) -> SuccessorSnapshot:
        row = self._m[state_id]
        total = sum(max(0.0, value) for value in row)
        if total <= 0:
            return SuccessorSnapshot(state_id=state_id, uncertainty=1.0, top_successors=())
        probs = [max(0.0, value) / total for value in row]
        ranked = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)[:top_k]
        uncertainty = 1.0 - (ranked[0][1] if ranked else 0.0)
        return SuccessorSnapshot(
            state_id=state_id,
            uncertainty=uncertainty,
            top_successors=tuple((idx, prob) for idx, prob in ranked),
        )
