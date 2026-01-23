# Unit tests for metadata stores.

from __future__ import annotations

from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.models import Chunk, Document


def test_inmemory_metadata_store_round_trip() -> None:
    store = InMemoryMetadataStore()
    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Doc",
        origin="/tmp/doc.md",
    )
    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Sample",
        chunk_index=1,
    )
    store.upsert_document(document)
    store.upsert_chunk(chunk)

    assert store.get_document("doc1") == document
    assert store.get_chunk("doc1#0001") == chunk
