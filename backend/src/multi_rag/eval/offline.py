# Offline evaluation utilities for retrieval and citation quality.

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol

from multi_rag.answering.pipeline import AnsweringPipeline
from multi_rag.indexing.metadata_store import InMemoryMetadataStore, PostgresMetadataStore
from multi_rag.models import AnswerResponse, Chunk
from multi_rag.retrieval.hybrid import HybridRetriever, RetrievalResult


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_sources: list[str] = field(default_factory=list)
    expected_refusal: bool = False
    notes: str | None = None


@dataclass
class EvalCaseResult:
    case_id: str
    query: str
    retrieval_hit: bool | None
    refusal_correct: bool
    citation_coverage: float
    claim_count: int
    citation_count: int
    refused: bool


@dataclass
class EvalMetrics:
    total_cases: int
    cases_with_expected_sources: int
    retrieval_hit_rate: float
    refusal_accuracy: float
    citation_coverage: float
    answered_rate: float
    avg_claims_per_answer: float
    avg_citations_per_answer: float


@dataclass
class EvalReport:
    metrics: EvalMetrics
    cases: list[EvalCaseResult]
    failures: list[str]


class PipelineRunner(Protocol):
    def answer(self, query: str, *, metadata_filter: dict | None = None) -> AnswerResponse:
        ...


def load_gold_set(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    cases: list[EvalCase] = []
    if not path.exists():
        raise FileNotFoundError(f"Gold set file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}") from exc
            case_id = str(payload.get("id") or payload.get("case_id") or "").strip()
            query = str(payload.get("query") or "").strip()
            if not case_id or not query:
                raise ValueError(f"Missing id/query on line {line_num}")
            expected_sources = payload.get("expected_sources") or []
            if isinstance(expected_sources, str):
                expected_sources = [expected_sources]
            if not isinstance(expected_sources, list):
                raise ValueError(f"expected_sources must be a list on line {line_num}")
            normalized_sources = [str(item).strip() for item in expected_sources if str(item).strip()]
            cases.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    expected_sources=normalized_sources,
                    expected_refusal=bool(payload.get("expected_refusal", False)),
                    notes=str(payload.get("notes")) if payload.get("notes") else None,
                )
            )
    return cases


def evaluate_gold_set(
    cases: list[EvalCase],
    *,
    retriever: HybridRetriever,
    pipeline: AnsweringPipeline | PipelineRunner,
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
    top_k: int = 5,
    thresholds: dict | None = None,
) -> EvalReport:
    thresholds = thresholds or {}
    case_results: list[EvalCaseResult] = []
    retrieval_hits = 0
    retrieval_total = 0
    refusal_hits = 0
    citation_total = 0.0
    answered = 0
    claims_total = 0
    citations_total = 0

    for case in cases:
        results = retriever.retrieve(case.query, top_k=top_k)
        response = pipeline.answer(case.query)
        retrieval_hit = _evaluate_retrieval_hit(case, results, metadata_store)
        if retrieval_hit is not None:
            retrieval_total += 1
            retrieval_hits += int(retrieval_hit)
        refusal_correct = response.refused == case.expected_refusal
        refusal_hits += int(refusal_correct)
        claim_count = len(response.claims)
        citation_count = len(response.citations)
        citations_total += citation_count
        claims_total += claim_count
        if not response.refused:
            answered += 1
        coverage = _citation_coverage(response)
        citation_total += coverage
        case_results.append(
            EvalCaseResult(
                case_id=case.case_id,
                query=case.query,
                retrieval_hit=retrieval_hit,
                refusal_correct=refusal_correct,
                citation_coverage=coverage,
                claim_count=claim_count,
                citation_count=citation_count,
                refused=response.refused,
            )
        )

    total_cases = len(cases)
    retrieval_hit_rate = retrieval_hits / retrieval_total if retrieval_total else 0.0
    refusal_accuracy = refusal_hits / total_cases if total_cases else 0.0
    citation_coverage = citation_total / total_cases if total_cases else 0.0
    answered_rate = answered / total_cases if total_cases else 0.0
    avg_claims = claims_total / answered if answered else 0.0
    avg_citations = citations_total / answered if answered else 0.0
    metrics = EvalMetrics(
        total_cases=total_cases,
        cases_with_expected_sources=retrieval_total,
        retrieval_hit_rate=retrieval_hit_rate,
        refusal_accuracy=refusal_accuracy,
        citation_coverage=citation_coverage,
        answered_rate=answered_rate,
        avg_claims_per_answer=avg_claims,
        avg_citations_per_answer=avg_citations,
    )
    failures = _check_thresholds(metrics, thresholds)
    return EvalReport(metrics=metrics, cases=case_results, failures=failures)


def _evaluate_retrieval_hit(
    case: EvalCase,
    results: list[RetrievalResult],
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
) -> bool | None:
    if not case.expected_sources:
        return None
    expected = [_normalize_source(source) for source in case.expected_sources]
    for result in results:
        chunk = metadata_store.get_chunk(result.chunk_id)
        if not chunk:
            continue
        if _matches_expected(chunk, metadata_store, expected):
            return True
    return False


def _matches_expected(
    chunk: Chunk,
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
    expected: list[str],
) -> bool:
    candidates = _collect_sources(chunk, metadata_store)
    for expected_source in expected:
        for candidate in candidates:
            if _source_match(candidate, expected_source):
                return True
    return False


def _collect_sources(
    chunk: Chunk,
    metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
) -> list[str]:
    document = metadata_store.get_document(chunk.doc_id)
    candidates = [
        chunk.doc_id,
        chunk.metadata.get("origin", ""),
        chunk.metadata.get("title", ""),
    ]
    if document:
        candidates.extend([document.origin, document.title, document.doc_id])
    return [item for item in candidates if item]


def _source_match(candidate: str, expected: str) -> bool:
    candidate_norm = _normalize_source(candidate)
    expected_norm = _normalize_source(expected)
    if not candidate_norm or not expected_norm:
        return False
    if candidate_norm == expected_norm:
        return True
    return expected_norm in candidate_norm


def _normalize_source(value: str) -> str:
    return value.strip().lower()


def _citation_coverage(response: AnswerResponse) -> float:
    if not response.claims:
        return 0.0
    with_citations = sum(1 for claim in response.claims if claim.citations)
    return with_citations / len(response.claims)


def _check_thresholds(metrics: EvalMetrics, thresholds: dict) -> list[str]:
    failures: list[str] = []
    minimum_hit = float(thresholds.get("min_retrieval_hit_rate", 0.0))
    if metrics.retrieval_hit_rate < minimum_hit:
        failures.append(
            f"Retrieval hit rate {metrics.retrieval_hit_rate:.2f} < {minimum_hit:.2f}"
        )
    minimum_citation = float(thresholds.get("min_citation_coverage", 0.0))
    if metrics.citation_coverage < minimum_citation:
        failures.append(
            f"Citation coverage {metrics.citation_coverage:.2f} < {minimum_citation:.2f}"
        )
    minimum_refusal = float(thresholds.get("min_refusal_accuracy", 0.0))
    if metrics.refusal_accuracy < minimum_refusal:
        failures.append(
            f"Refusal accuracy {metrics.refusal_accuracy:.2f} < {minimum_refusal:.2f}"
        )
    return failures
