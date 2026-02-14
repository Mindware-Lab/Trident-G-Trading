from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Signal:
    side: str
    strength: float


class Operator(Protocol):
    name: str

    def on_event(self, event: dict[str, object]) -> Signal | None: ...
