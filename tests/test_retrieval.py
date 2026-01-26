# Unit tests for hybrid retrieval.

from __future__ import annotations

from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord
from multi_rag.models import Chunk, Document
from multi_rag.retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_returns_matching_chunk() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vector_store = InMemoryVectorStore()
    keyword_index = BM25Index()
    metadata_store = InMemoryMetadataStore()

    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Doc",
        origin="/tmp/doc.md",
    )
    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days.",
        chunk_index=1,
        metadata={"source_type": "markdown"},
    )
    metadata_store.upsert_document(document)
    metadata_store.upsert_chunk(chunk)

    vector = embedder.embed_texts([chunk.chunk_text])[0]
    vector_store.upsert([VectorRecord(record_id=chunk.chunk_id, vector=vector, payload=chunk.metadata)])
    keyword_index.add_documents([chunk.chunk_id], [chunk.chunk_text])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
    )
    results = retriever.retrieve("retention logs", top_k=1)
    assert results[0].chunk_id == "doc1#0001"
