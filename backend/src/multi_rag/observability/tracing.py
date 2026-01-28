# Lightweight tracing hooks for retrieval and answering.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from typing import Protocol
import uuid


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


class StructuredLogTracer:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        service_name: str = "multi-rag",
        trace_id: str | None = None,
        static_fields: dict | None = None,
    ) -> None:
        self._logger = logger
        self._service_name = service_name
        self._trace_id = trace_id or uuid.uuid4().hex
        self._static_fields = static_fields or {}

    def record_event(self, name: str, payload: dict) -> None:
        event = {
            "event": name,
            "service": self._service_name,
            "trace_id": payload.get("trace_id") if isinstance(payload, dict) else None,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "payload": payload,
        }
        event["trace_id"] = event["trace_id"] or self._trace_id
        if self._static_fields:
            event.update(self._static_fields)
        self._logger.info(json.dumps(event, separators=(",", ":"), sort_keys=True))


def build_tracer(
    mode: str,
    *,
    logger_name: str = "multi_rag.observability",
    service_name: str = "multi-rag",
    static_fields: dict | None = None,
) -> Tracer:
    normalized = (mode or "").strip().lower()
    if normalized in {"structured", "json", "log", "logs"}:
        logger = logging.getLogger(logger_name)
        return StructuredLogTracer(
            logger=logger, service_name=service_name, static_fields=static_fields
        )
    if normalized in {"memory", "in-memory", "in_memory"}:
        return InMemoryTracer()
    return NullTracer()
