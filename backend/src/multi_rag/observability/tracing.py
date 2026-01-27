# Lightweight tracing hooks for retrieval and answering.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class TraceEvent:
    name: str
    payload: dict
    recorded_at: datetime


class Tracer(Protocol):
    def record_event(self, name: str, payload: dict) -> None:
        ...


class NullTracer:
    def record_event(self, name: str, payload: dict) -> None:
        return None


class InMemoryTracer:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record_event(self, name: str, payload: dict) -> None:
        self._events.append(
            TraceEvent(name=name, payload=payload, recorded_at=datetime.utcnow())
        )

    def list_events(self) -> list[TraceEvent]:
        return list(self._events)
