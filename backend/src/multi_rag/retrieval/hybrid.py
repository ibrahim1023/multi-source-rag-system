# Hybrid retrieval and reranking.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import EmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore, PostgresMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore
from multi_rag.models import Chunk
from multi_rag.retrieval.query_expansion import QueryExpansionConfig, expand_query
from multi_rag.retrieval.reranker import Reranker


@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    payload: dict


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: InMemoryVectorStore,
        keyword_index: BM25Index,
        metadata_store: InMemoryMetadataStore | PostgresMetadataStore,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        query_config: QueryExpansionConfig | None = None,
        reranker: Reranker | None = None,
        rerank_top_k: int = 8,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._metadata_store = metadata_store
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._query_config = query_config or QueryExpansionConfig()
        self._reranker = reranker
        self._rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        expanded = expand_query(query, self._query_config)
        vector_query = query
        keyword_query = query
        if self._query_config.use_expanded_for_embeddings:
            vector_query = expanded.expanded_text
        if self._query_config.use_expanded_for_bm25:
            keyword_query = expanded.expanded_text
        vector_results = self._vector_store.search_with_scores(
            self._embedder.embed_texts([vector_query])[0],
            limit=top_k * 2,
        )
        keyword_results = self._keyword_index.search(keyword_query, top_k=top_k * 2)

        vector_scores = {record.record_id: score for record,
                         score in vector_results}
        keyword_scores = {chunk_id: score for chunk_id,
                          score in keyword_results}

        all_ids = set(vector_scores) | set(keyword_scores)
        scored: list[RetrievalResult] = []
        for chunk_id in all_ids:
            chunk = self._metadata_store.get_chunk(chunk_id)
            if not chunk:
                continue
            if metadata_filter and not _matches_filter(chunk.metadata, metadata_filter):
                continue
            score = (
                self._vector_weight * vector_scores.get(chunk_id, 0.0)
                + self._keyword_weight * keyword_scores.get(chunk_id, 0.0)
            )
            scored.append(RetrievalResult(chunk_id=chunk_id,
                          score=score, payload=chunk.metadata))

        scored.sort(key=lambda item: item.score, reverse=True)
        scored = self._apply_reranker(query, scored)
        return scored[:top_k]

    def assemble_context(
        self,
        results: list[RetrievalResult],
        *,
        max_chunks: int = 6,
        neighbor_window: int = 1,
        focus_top_doc: bool = False,
    ) -> list[Chunk]:
        context: list[Chunk] = []
        seen: set[str] = set()
        focus_doc_id = None
        if focus_top_doc:
            for result in results:
                candidate = self._metadata_store.get_chunk(result.chunk_id)
                if candidate:
                    focus_doc_id = candidate.doc_id
                    break
        for result in results:
            if len(context) >= max_chunks:
                break
            chunk = self._metadata_store.get_chunk(result.chunk_id)
            if not chunk:
                continue
            if focus_doc_id and chunk.doc_id != focus_doc_id:
                continue
            for candidate in self._expand_section_window(chunk, neighbor_window):
                if candidate.chunk_id in seen:
                    continue
                seen.add(candidate.chunk_id)
                context.append(candidate)
                if len(context) >= max_chunks:
                    break
        return context

    def _expand_section_window(self, chunk: Chunk, neighbor_window: int) -> list[Chunk]:
        if neighbor_window <= 0:
            return [chunk]

        chunks = self._metadata_store.list_chunks(chunk.doc_id)
        if not chunks:
            return [chunk]

        target_index = None
        for idx, candidate in enumerate(chunks):
            if candidate.chunk_id == chunk.chunk_id:
                target_index = idx
                break
        if target_index is None:
            return [chunk]

        start = max(target_index - neighbor_window, 0)
        end = min(target_index + neighbor_window, len(chunks) - 1)
        window = chunks[start:end + 1]
        if chunk.section_path:
            window = [item for item in window if item.section_path == chunk.section_path]
        return window

    def _apply_reranker(
        self, query: str, scored: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        if not scored or not self._reranker:
            return scored
        limit = min(self._rerank_top_k, len(scored))
        candidates = [
            self._metadata_store.get_chunk(result.chunk_id)
            for result in scored[:limit]
        ]
        chunks = [chunk for chunk in candidates if chunk]
        if not chunks:
            return scored
        reranked = self._reranker.rerank(query, chunks)
        if not reranked:
            return scored
        original_map = {result.chunk_id: result for result in scored}
        reranked_results: list[RetrievalResult] = []
        used: set[str] = set()
        for chunk_id, score in reranked:
            original = original_map.get(chunk_id)
            if not original:
                continue
            reranked_results.append(
                RetrievalResult(chunk_id=chunk_id, score=score, payload=original.payload)
            )
            used.add(chunk_id)
        for result in scored:
            if result.chunk_id in used:
                continue
            reranked_results.append(result)
        return reranked_results


def _matches_filter(metadata: dict, filters: dict) -> bool:
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True
