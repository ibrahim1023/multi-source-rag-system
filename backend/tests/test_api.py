# API endpoint tests.

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

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

    status_response = client.get("/ingest/status")
    assert status_response.status_code == 200
    jobs = status_response.json()["jobs"]
    assert jobs
    assert jobs[0]["status"] == "completed"

    search_response = client.post("/search", json={"query": "retention logs", "top_k": 1})
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert results

    chat_response = client.post("/chat", json={"query": "retention logs"})
    assert chat_response.status_code == 200
    chat_body = chat_response.json()
    assert chat_body["refused"] is False
    assert chat_body["citations"]

    documents_response = client.get("/documents")
    assert documents_response.status_code == 200
    documents = documents_response.json()["documents"]
    assert documents


def test_ingest_file_flow() -> None:
    settings = APISettings(metadata_backend="memory", embedding_provider="hash", embedding_dim=8)
    dependencies = build_dependencies(settings)
    app = create_app(settings=settings, dependencies=dependencies)

    client = TestClient(app)
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 20 100 Td (Retention policy for logs is 30 days.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    pdf_bytes = buffer.getvalue()
    data = {
        "title": "Doc",
        "origin": "/tmp/doc.pdf",
        "metadata": '{"tags": ["policy"]}',
    }
    files = {"file": ("doc.pdf", pdf_bytes)}
    response = client.post("/ingest/pdf/file", data=data, files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks_indexed"] == 1
