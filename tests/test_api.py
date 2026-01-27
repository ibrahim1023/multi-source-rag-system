# API endpoint tests.

from __future__ import annotations

from fastapi.testclient import TestClient

from multi_rag.api.app import APISettings, build_dependencies, create_app


def test_ingest_search_and_chat_flow() -> None:
    settings = APISettings(metadata_backend="memory", embedding_provider="hash", embedding_dim=8)
    dependencies = build_dependencies(settings)
    app = create_app(settings=settings, dependencies=dependencies)

    client = TestClient(app)
    ingest_payload = {
        "title": "Doc",
        "origin": "/tmp/doc.md",
        "text": "Retention policy for logs is 30 days.",
        "metadata": {"tags": ["policy"]},
    }
    response = client.post("/ingest/markdown", json=ingest_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] == 1

    search_response = client.post("/search", json={"query": "retention logs", "top_k": 1})
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert results

    chat_response = client.post("/chat", json={"query": "retention logs"})
    assert chat_response.status_code == 200
    chat_body = chat_response.json()
    assert chat_body["refused"] is False
    assert chat_body["citations"]
