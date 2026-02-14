from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyContext:
    regime_label: str
    zone: str
    explore_pressure: float


def select_operator(context: PolicyContext, candidate_scores: dict[str, float]) -> str:
    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty")

    adjusted = dict(candidate_scores)
    if context.regime_label == "shock":
        adjusted["breakout"] = adjusted.get("breakout", 0.0) + 0.15
    elif context.regime_label == "calm":
        adjusted["mean_reversion"] = adjusted.get("mean_reversion", 0.0) + 0.10

    if context.zone == "reset":
        adjusted["flat"] = adjusted.get("flat", 0.0) + 1.0

    if context.explore_pressure > 0.6:
        adjusted = {name: score + 0.05 for name, score in adjusted.items()}

    return max(adjusted, key=lambda name: adjusted[name])
