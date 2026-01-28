# Tests for observability tracing utilities.

from __future__ import annotations

import logging

from multi_rag.observability.tracing import StructuredLogTracer, build_tracer


def test_build_tracer_structured_emits_json(caplog) -> None:
    tracer = build_tracer(
        "structured",
        logger_name="test.observability",
        service_name="test-service",
        static_fields={"env": "test"},
    )
    assert isinstance(tracer, StructuredLogTracer)
    with caplog.at_level(logging.INFO, logger="test.observability"):
        tracer.record_event("test.event", {"value": 1})
    assert any("\"event\":\"test.event\"" in record.message for record in caplog.records)
