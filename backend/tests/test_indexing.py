# Unit tests for embedding and vector storage helpers.

from __future__ import annotations

from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord


def test_inmemory_vector_store_retrieves_nearest_neighbor() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vectors = embedder.embed_texts(["alpha", "beta"])
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(record_id="a", vector=vectors[0], payload={"text": "alpha"}),
            VectorRecord(record_id="b", vector=vectors[1], payload={"text": "beta"}),
        ]
    )

    query = embedder.embed_texts(["alpha"])[0]
    results = store.search(query, limit=1)
    assert results[0].record_id == "a"


def test_inmemory_vector_store_deletes_records() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vectors = embedder.embed_texts(["alpha", "beta"])
    store = InMemoryVectorStore()
    store.upsert(
        [
            VectorRecord(record_id="a", vector=vectors[0], payload={"text": "alpha"}),
            VectorRecord(record_id="b", vector=vectors[1], payload={"text": "beta"}),
        ]
    )

    store.delete_by_ids({"a"})
    query = embedder.embed_texts(["alpha"])[0]
    results = store.search(query, limit=2)
    assert all(record.record_id != "a" for record in results)
