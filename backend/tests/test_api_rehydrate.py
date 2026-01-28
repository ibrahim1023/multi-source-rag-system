# Tests for API index rehydration behavior.

from __future__ import annotations

from multi_rag.api.app import APISettings, _rehydrate_indexes, build_dependencies
from multi_rag.models import Chunk, Document


def test_rehydrate_indexes_populates_retriever() -> None:
    settings = APISettings()
    deps = build_dependencies(settings)

    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Retention",
        origin="/docs/retention.md",
    )
    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days.",
        chunk_index=1,
        metadata={},
    )
    deps.metadata_store.upsert_document(document)
    deps.metadata_store.upsert_chunk(chunk)

    _rehydrate_indexes(deps)

    results = deps.retriever.retrieve("retention")
    assert results
