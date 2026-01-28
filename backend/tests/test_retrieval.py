# Unit tests for hybrid retrieval.

from __future__ import annotations

from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord
from multi_rag.models import Chunk, Document
from multi_rag.retrieval.hybrid import HybridRetriever, RetrievalResult
from multi_rag.retrieval.query_expansion import QueryExpansionConfig, expand_query


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


def test_assemble_context_adds_neighbors_and_dedupes() -> None:
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
    metadata_store.upsert_document(document)

    chunks = [
        Chunk(
            chunk_id="doc1#0001",
            doc_id="doc1",
            chunk_text="Section one content.",
            chunk_index=1,
            section_path="Section One",
            metadata={"source_type": "markdown"},
        ),
        Chunk(
            chunk_id="doc1#0002",
            doc_id="doc1",
            chunk_text="More details in section one.",
            chunk_index=2,
            section_path="Section One",
            metadata={"source_type": "markdown"},
        ),
        Chunk(
            chunk_id="doc1#0003",
            doc_id="doc1",
            chunk_text="Final note in section one.",
            chunk_index=3,
            section_path="Section One",
            metadata={"source_type": "markdown"},
        ),
    ]
    for chunk in chunks:
        metadata_store.upsert_chunk(chunk)

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
    )
    results = [
        RetrievalResult(chunk_id="doc1#0002", score=1.0, payload={}),
        RetrievalResult(chunk_id="doc1#0003", score=0.9, payload={}),
    ]
    context = retriever.assemble_context(results, max_chunks=3, neighbor_window=1)
    assert [chunk.chunk_id for chunk in context] == [
        "doc1#0001",
        "doc1#0002",
        "doc1#0003",
    ]


def test_query_expansion_removes_stopwords_and_adds_variants() -> None:
    config = QueryExpansionConfig()
    expanded = expand_query("the retention policies", config)
    assert "the" not in expanded.keywords
    assert "policy" in expanded.expanded_tokens
    assert "policies" in expanded.expanded_tokens


def test_reranker_overrides_order() -> None:
    class DummyReranker:
        def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[str, float]]:
            ordered = sorted(chunks, key=lambda item: item.chunk_id, reverse=True)
            return [(ordered[0].chunk_id, 1.0), (ordered[1].chunk_id, 0.5)]

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
    metadata_store.upsert_document(document)

    chunks = [
        Chunk(
            chunk_id="doc1#0001",
            doc_id="doc1",
            chunk_text="First chunk text.",
            chunk_index=1,
            metadata={"source_type": "markdown"},
        ),
        Chunk(
            chunk_id="doc1#0002",
            doc_id="doc1",
            chunk_text="Second chunk text.",
            chunk_index=2,
            metadata={"source_type": "markdown"},
        ),
    ]
    for chunk in chunks:
        metadata_store.upsert_chunk(chunk)
        vector = embedder.embed_texts([chunk.chunk_text])[0]
        vector_store.upsert([VectorRecord(record_id=chunk.chunk_id, vector=vector, payload=chunk.metadata)])
        keyword_index.add_documents([chunk.chunk_id], [chunk.chunk_text])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
        reranker=DummyReranker(),
        rerank_top_k=2,
    )
    results = retriever.retrieve("chunk text", top_k=2)
    assert results[0].chunk_id == "doc1#0002"


def test_freshness_weighting_boosts_recent_sources() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vector_store = InMemoryVectorStore()
    keyword_index = BM25Index()
    metadata_store = InMemoryMetadataStore()

    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Doc",
        origin="/tmp/doc.md",
        updated_at=None,
    )
    metadata_store.upsert_document(document)

    chunks = [
        Chunk(
            chunk_id="doc1#0001",
            doc_id="doc1",
            chunk_text="Retention policy for logs.",
            chunk_index=1,
            metadata={"source_type": "markdown", "updated_at": "2000-01-01T00:00:00"},
        ),
        Chunk(
            chunk_id="doc1#0002",
            doc_id="doc1",
            chunk_text="Retention policy for logs.",
            chunk_index=2,
            metadata={"source_type": "markdown", "updated_at": "2025-01-01T00:00:00"},
        ),
    ]
    for chunk in chunks:
        metadata_store.upsert_chunk(chunk)
        vector = embedder.embed_texts([chunk.chunk_text])[0]
        vector_store.upsert([VectorRecord(record_id=chunk.chunk_id, vector=vector, payload=chunk.metadata)])
        keyword_index.add_documents([chunk.chunk_id], [chunk.chunk_text])

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        metadata_store=metadata_store,
        freshness_enabled=True,
        freshness_weight=1.0,
        freshness_half_life_days=3650.0,
    )
    results = retriever.retrieve("retention policy", top_k=2)
    assert results[0].chunk_id == "doc1#0002"
