# Optional rerankers for top retrieval candidates.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.models import Chunk


class Reranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[str, float]]:
        raise NotImplementedError


@dataclass
class CrossEncoderReranker(Reranker):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "sentence-transformers is required for CrossEncoderReranker."
            ) from exc
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[str, float]]:
        if not chunks:
            return []
        pairs = [(query, chunk.chunk_text) for chunk in chunks]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        scored = [
            (chunk.chunk_id, float(score)) for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored
