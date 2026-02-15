from __future__ import annotations

import random
from dataclasses import dataclass, field

from trident_trader.features.mutual_info import rolling_mi_relevance
from trident_trader.features.policy_entropy import normalized_entropy, softmax, update_temperature


@dataclass(frozen=True)
class SelectorConfig:
    operators: tuple[str, ...] = ("mean_reversion", "breakout")
    entropy_target: float = 0.72
    tau_init: float = 1.0
    tau_min: float = 0.3
    tau_max: float = 3.0
    tau_step: float = 0.2
    mi_min: float = 0.02
    mi_window: int = 240
    mi_n_min: int = 200
    mismatch_alert: float = 0.15
    sr_uncertainty_weight: float = 0.6


@dataclass
class OperatorSelectorEntropyMI:
    config: SelectorConfig = field(default_factory=SelectorConfig)
    seed: int = 11
    _q: dict[str, float] = field(default_factory=dict)
    _tau: float = 1.0
    _features_hist: list[list[float]] = field(default_factory=list)
    _reward_hist: list[float] = field(default_factory=list)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if not self._q:
            self._q = {op: 0.0 for op in self.config.operators}
        self._tau = self.config.tau_init

    @property
    def temperature(self) -> float:
        return self._tau

    def observe(self, operator: str, reward: float, feature_vector: list[float]) -> None:
        if operator in self._q:
            self._q[operator] = 0.9 * self._q[operator] + 0.1 * reward
        self._features_hist.append(feature_vector)
        self._reward_hist.append(reward)

    def select(
        self,
        armed: bool,
        mismatch: float,
        feature_vector: list[float],
        sr_uncertainty: float = 0.0,
    ) -> tuple[str, float]:
        if not armed:
            return "flat", 0.0

        scores = [self._q[op] for op in self.config.operators]
        probs = softmax(scores, self._tau)
        entropy = normalized_entropy(probs)

        self._tau = update_temperature(
            current_tau=self._tau,
            entropy_value=entropy,
            entropy_target=self.config.entropy_target,
            step=self.config.tau_step,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
        )

        mi_result = rolling_mi_relevance(
            features_history=self._features_hist + [feature_vector],
            rewards_history=self._reward_hist + [0.0],
            window=self.config.mi_window,
            n_min=self.config.mi_n_min,
        )
        if (
            not mi_result.stable
            or mi_result.value < self.config.mi_min
            or mismatch > self.config.mismatch_alert
        ):
            self._tau = min(self.config.tau_max, self._tau + self.config.tau_step)
        else:
            self._tau = max(self.config.tau_min, self._tau - 0.5 * self.config.tau_step)

        # Map-level uncertainty increases exploration pressure.
        self._tau = min(
            self.config.tau_max,
            self._tau + self.config.sr_uncertainty_weight * max(0.0, sr_uncertainty),
        )

        probs = softmax(scores, self._tau)
        index = self._rng.choices(range(len(self.config.operators)), weights=probs, k=1)[0]
        op = self.config.operators[index]
        return op, mi_result.value
