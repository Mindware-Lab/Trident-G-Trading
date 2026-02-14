from __future__ import annotations

from dataclasses import dataclass, field

from trident_trader.features.policy_entropy import normalized_entropy, softmax, update_temperature


@dataclass(frozen=True)
class EntropyMIConfig:
    entropy_target: float = 0.72
    temperature_init: float = 1.0
    temperature_step: float = 0.25
    temperature_min: float = 0.3
    temperature_max: float = 3.0
    mi_floor: float = 0.05
    mismatch_alert: float = 0.18


@dataclass(frozen=True)
class ControllerInputs:
    policy_scores: list[float]
    signal_mi: float
    operator_mi: float
    mismatch_value: float
    lambda_world: float


@dataclass(frozen=True)
class ControllerOutput:
    mode: str
    explore_pressure: float
    exploit_pressure: float
    temperature: float
    policy_entropy: float
    mi_score: float


@dataclass
class EntropyMIController:
    config: EntropyMIConfig = field(default_factory=EntropyMIConfig)
    temperature: float = field(default=1.0)

    def __post_init__(self) -> None:
        if self.temperature == 1.0:
            self.temperature = self.config.temperature_init

    def step(self, inputs: ControllerInputs) -> ControllerOutput:
        probs = softmax(inputs.policy_scores, self.temperature)
        h_policy = normalized_entropy(probs)
        self.temperature = update_temperature(
            current_tau=self.temperature,
            entropy_value=h_policy,
            entropy_target=self.config.entropy_target,
            step=self.config.temperature_step,
            tau_min=self.config.temperature_min,
            tau_max=self.config.temperature_max,
        )

        mi_score = max(0.0, 0.6 * inputs.signal_mi + 0.4 * inputs.operator_mi)
        mismatch_pressure = min(1.0, max(0.0, abs(inputs.mismatch_value) / 0.5))
        mi_deficit = min(
            1.0, max(0.0, (self.config.mi_floor - mi_score) / max(self.config.mi_floor, 1e-6))
        )

        explore_pressure = min(
            1.0,
            0.45 * (self.config.entropy_target - h_policy + 1.0) / 2.0
            + 0.35 * mismatch_pressure
            + 0.20 * mi_deficit,
        )
        exploit_pressure = min(
            1.0, max(0.0, 0.6 * mi_score + 0.3 * inputs.lambda_world - 0.3 * mismatch_pressure)
        )

        mode = "exploit"
        if (
            mi_score < self.config.mi_floor
            or abs(inputs.mismatch_value) > self.config.mismatch_alert
        ):
            mode = "explore"

        return ControllerOutput(
            mode=mode,
            explore_pressure=explore_pressure,
            exploit_pressure=exploit_pressure,
            temperature=self.temperature,
            policy_entropy=h_policy,
            mi_score=mi_score,
        )
