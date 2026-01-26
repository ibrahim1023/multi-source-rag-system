# Context-only answering with citations and confidence policy.

from __future__ import annotations

from dataclasses import dataclass
import re

from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.models import AnswerResponse, Claim, Citation, Chunk
from multi_rag.retrieval.hybrid import RetrievalResult


@dataclass
class AnsweringConfig:
    max_claims: int = 3
    max_snippet_chars: int = 240
    high_confidence: float = 0.7
    medium_confidence: float = 0.45


class GroundedAnswerer:
    def __init__(
        self,
        *,
        metadata_store: InMemoryMetadataStore,
        config: AnsweringConfig | None = None,
    ) -> None:
        self._metadata_store = metadata_store
        self._config = config or AnsweringConfig()

    def answer(
        self,
        query: str,
        results: list[RetrievalResult],
        context: list[Chunk],
    ) -> AnswerResponse:
        if not results or not context:
            return self._refusal_response(query, confidence=0.0)

        query_terms = _tokenize(query)
        score_map = {result.chunk_id: result.score for result in results}
        candidates = _collect_candidates(context, query_terms, score_map)
        if not candidates:
            return self._refusal_response(query, confidence=0.0)

        claims = self._select_claims(candidates)
        confidence = _score_confidence(query_terms, results, claims)
        mode = _confidence_mode(confidence, self._config)
        if mode == "low":
            return self._refusal_response(query, confidence=confidence)

        citations = _dedupe_citations([citation for claim in claims for citation in claim.citations])
        answer_text = " ".join([claim.text for claim in claims]).strip()
        follow_up = None
        if mode == "medium":
            follow_up = _clarifying_question(query_terms)
        return AnswerResponse(
            answer=answer_text,
            claims=claims,
            citations=citations,
            confidence=confidence,
            mode=mode,
            follow_up_question=follow_up,
            refused=False,
        )

    def _select_claims(self, candidates: list[tuple[float, int, int, Chunk, str]]) -> list[Claim]:
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[: self._config.max_claims]
        selected.sort(key=lambda item: (item[1], item[2]))
        seen: set[str] = set()
        claims: list[Claim] = []
        for _, _, _, chunk, sentence in selected:
            normalized = sentence.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            citation = _build_citation(self._metadata_store, chunk, sentence, self._config)
            claims.append(Claim(text=normalized, citations=[citation]))
        return claims

    def _refusal_response(self, query: str, *, confidence: float) -> AnswerResponse:
        return AnswerResponse(
            answer="I do not have enough context to answer that yet.",
            claims=[],
            citations=[],
            confidence=confidence,
            mode="low",
            follow_up_question=_clarifying_question(_tokenize(query)),
            refused=True,
        )


def _collect_candidates(
    context: list[Chunk],
    query_terms: list[str],
    score_map: dict[str, float],
) -> list[tuple[float, int, int, Chunk, str]]:
    candidates: list[tuple[float, int, int, Chunk, str]] = []
    matching_found = False
    for chunk in context:
        sentences = _split_sentences(chunk.chunk_text)
        for idx, sentence in enumerate(sentences):
            coverage = _term_coverage(sentence, query_terms)
            if coverage > 0:
                matching_found = True
            score = score_map.get(chunk.chunk_id, 0.0) * 0.7 + coverage * 0.3
            candidates.append((score, chunk.chunk_index, idx, chunk, sentence))

    if matching_found:
        return [item for item in candidates if _term_coverage(item[4], query_terms) > 0]
    return candidates


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _term_coverage(sentence: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0
    sentence_terms = set(_tokenize(sentence))
    if not sentence_terms:
        return 0.0
    matched = len(set(query_terms) & sentence_terms)
    return matched / len(set(query_terms))


def _score_confidence(
    query_terms: list[str],
    results: list[RetrievalResult],
    claims: list[Claim],
) -> float:
    if not results or not claims:
        return 0.0
    top_score = results[0].score
    second_score = results[1].score if len(results) > 1 else 0.0
    spread = max(0.0, top_score - second_score)
    normalized_spread = spread / max(abs(top_score), 1e-6)
    normalized_spread = max(0.0, min(1.0, normalized_spread))

    coverage = 0.5
    if query_terms:
        claim_terms = set(_tokenize(" ".join([claim.text for claim in claims])))
        coverage = len(set(query_terms) & claim_terms) / len(set(query_terms))

    citation_coverage = sum(1 for claim in claims if claim.citations) / len(claims)
    score = 0.4 * coverage + 0.3 * citation_coverage + 0.3 * normalized_spread
    return max(0.0, min(1.0, score))


def _confidence_mode(confidence: float, config: AnsweringConfig) -> str:
    if confidence >= config.high_confidence:
        return "high"
    if confidence >= config.medium_confidence:
        return "medium"
    return "low"


def _clarifying_question(query_terms: list[str]) -> str:
    if query_terms:
        return "Which source or time range should I focus on?"
    return "Can you share more detail about what you need?"


def _build_citation(
    metadata_store: InMemoryMetadataStore,
    chunk: Chunk,
    sentence: str,
    config: AnsweringConfig,
) -> Citation:
    document = metadata_store.get_document(chunk.doc_id)
    title = document.title if document else chunk.metadata.get("title", chunk.doc_id)
    origin = document.origin if document else chunk.metadata.get("origin", "")
    snippet = sentence[: config.max_snippet_chars].strip()
    return Citation(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        title=title,
        origin=origin,
        section_path=chunk.section_path,
        snippet=snippet,
    )


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[str] = set()
    deduped: list[Citation] = []
    for citation in citations:
        if citation.chunk_id in seen:
            continue
        seen.add(citation.chunk_id)
        deduped.append(citation)
    return deduped
