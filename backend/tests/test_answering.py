# Unit tests for grounded answering.

from __future__ import annotations

from multi_rag.answering.grounded import GroundedAnswerer, _filter_orphan_claims
from multi_rag.answering.pipeline import AnsweringPipeline
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord
from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import HashEmbeddingProvider
from multi_rag.models import Chunk, Claim, Citation, Document
from multi_rag.retrieval.hybrid import HybridRetriever, RetrievalResult


def test_grounded_answerer_returns_claims_with_citations() -> None:
    metadata_store = InMemoryMetadataStore()
    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Policy",
        origin="/tmp/policy.md",
    )
    metadata_store.upsert_document(document)

    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days. Keep backups monthly.",
        chunk_index=1,
        section_path="Retention",
        metadata={"source_type": "markdown"},
    )
    metadata_store.upsert_chunk(chunk)

    answerer = GroundedAnswerer(metadata_store=metadata_store)
    results = [RetrievalResult(chunk_id="doc1#0001", score=0.9, payload={})]
    response = answerer.answer("retention logs", results, [chunk])

    assert response.refused is False
    assert "Retention policy for logs is 30 days." in response.answer
    assert response.claims
    assert response.citations
    assert response.citations[0].chunk_id == "doc1#0001"


def test_grounded_answerer_refuses_without_context() -> None:
    metadata_store = InMemoryMetadataStore()
    answerer = GroundedAnswerer(metadata_store=metadata_store)

    response = answerer.answer("retention logs", [], [])

    assert response.refused is True
    assert response.mode == "low"
    assert response.follow_up_question


def test_answering_pipeline_wires_retrieval_and_answering() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vector_store = InMemoryVectorStore()
    keyword_index = BM25Index()
    metadata_store = InMemoryMetadataStore()

    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Policy",
        origin="/tmp/policy.md",
    )
    metadata_store.upsert_document(document)

    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days.",
        chunk_index=1,
        section_path="Retention",
        metadata={"source_type": "markdown"},
    )
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
    answerer = GroundedAnswerer(metadata_store=metadata_store)
    pipeline = AnsweringPipeline(retriever=retriever, answerer=answerer)

    response = pipeline.answer("retention logs")

    assert response.refused is False
    assert response.citations


def test_pipeline_suppresses_follow_up_when_filtered() -> None:
    embedder = HashEmbeddingProvider(dim=8)
    vector_store = InMemoryVectorStore()
    keyword_index = BM25Index()
    metadata_store = InMemoryMetadataStore()

    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Policy",
        origin="/tmp/policy.md",
    )
    metadata_store.upsert_document(document)

    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days.",
        chunk_index=1,
        section_path="Retention",
        metadata={"source_type": "markdown", "origin": "/tmp/policy.md"},
    )
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
    answerer = GroundedAnswerer(metadata_store=metadata_store)
    pipeline = AnsweringPipeline(retriever=retriever, answerer=answerer)

    response = pipeline.answer("retention logs", metadata_filter={"origin": "/tmp/policy.md"})

    assert response.follow_up_question is None


def test_orphan_claims_are_filtered() -> None:
    context = [
        Chunk(
            chunk_id="doc1#0001",
            doc_id="doc1",
            chunk_text="Retention policy for logs is 30 days.",
            chunk_index=1,
            metadata={"source_type": "markdown"},
        )
    ]
    claims = [
        Claim(
            text="Retention policy for logs is 30 days.",
            citations=[Citation(doc_id="doc1", chunk_id="doc1#0001", title="Doc", origin="")],
        ),
        Claim(
            text="Uncited claim.",
            citations=[Citation(doc_id="doc1", chunk_id="doc1#9999", title="Doc", origin="")],
        ),
    ]

    filtered = _filter_orphan_claims(claims, context)

    assert len(filtered) == 1
    assert filtered[0].text == "Retention policy for logs is 30 days."
