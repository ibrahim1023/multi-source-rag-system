# Tests for offline evaluation helpers.

from __future__ import annotations

import json
from pathlib import Path

from multi_rag.eval.offline import EvalCase, evaluate_gold_set, load_gold_set
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.models import AnswerResponse, Chunk, Claim, Citation, Document
from multi_rag.retrieval.hybrid import RetrievalResult


class DummyRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, *, top_k: int = 5, metadata_filter: dict | None = None):
        return self._results[:top_k]


class DummyPipeline:
    def __init__(self, response: AnswerResponse) -> None:
        self._response = response

    def answer(self, query: str, *, metadata_filter: dict | None = None) -> AnswerResponse:
        return self._response


def test_load_gold_set(tmp_path: Path) -> None:
    payload = {
        "id": "case-1",
        "query": "What is retention?",
        "expected_sources": ["policy.pdf"],
        "expected_refusal": False,
    }
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cases = load_gold_set(path)
    assert len(cases) == 1
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_sources == ["policy.pdf"]


def test_evaluate_gold_set_metrics() -> None:
    metadata_store = InMemoryMetadataStore()
    document = Document(
        doc_id="doc-1",
        source_type="pdf",
        title="Retention Policy",
        origin="/docs/policy.pdf",
    )
    chunk = Chunk(
        chunk_id="doc-1#0001",
        doc_id="doc-1",
        chunk_text="Retention is 30 days.",
        chunk_index=0,
        metadata={"origin": document.origin, "title": document.title},
    )
    metadata_store.upsert_document(document)
    metadata_store.upsert_chunk(chunk)

    results = [RetrievalResult(chunk_id=chunk.chunk_id, score=1.0, payload={})]
    retriever = DummyRetriever(results)
    response = AnswerResponse(
        answer="Retention is 30 days.",
        claims=[
            Claim(
                text="Retention is 30 days.",
                citations=[
                    Citation(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        title=document.title,
                        origin=document.origin,
                    )
                ],
            )
        ],
        citations=[
            Citation(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                title=document.title,
                origin=document.origin,
            )
        ],
        confidence=0.8,
        mode="high",
        refused=False,
    )
    pipeline = DummyPipeline(response)
    cases = [EvalCase(case_id="case-1", query="retention", expected_sources=[document.origin])]

    report = evaluate_gold_set(
        cases,
        retriever=retriever,
        pipeline=pipeline,
        metadata_store=metadata_store,
        top_k=5,
        thresholds={"min_retrieval_hit_rate": 0.5},
    )

    assert report.metrics.retrieval_hit_rate == 1.0
    assert report.metrics.citation_coverage == 1.0
    assert not report.failures
