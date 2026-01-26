# Hybrid retrieval and reranking.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.indexing.bm25 import BM25Index
from multi_rag.indexing.embeddings import EmbeddingProvider
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.indexing.vector_store import InMemoryVectorStore, VectorRecord
from multi_rag.models import Chunk


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
        metadata_store: InMemoryMetadataStore,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._metadata_store = metadata_store
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[RetrievalResult]:
        vector_results = self._vector_store.search_with_scores(
            self._embedder.embed_texts([query])[0],
            limit=top_k * 2,
        )
        keyword_results = self._keyword_index.search(query, top_k=top_k * 2)

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
        return scored[:top_k]

    def assemble_context(
        self,
        results: list[RetrievalResult],
        *,
        max_chunks: int = 6,
        neighbor_window: int = 1,
    ) -> list[Chunk]:
        context: list[Chunk] = []
        seen: set[str] = set()
        for result in results:
            if len(context) >= max_chunks:
                break
            chunk = self._metadata_store.get_chunk(result.chunk_id)
            if not chunk:
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


def _matches_filter(metadata: dict, filters: dict) -> bool:
    for key, value in filters.items():
        if metadata.get(key) != value:
            return False
    return True
