# Vector store implementations.

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class VectorRecord:
    record_id: str
    vector: list[float]
    payload: dict


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def search(self, query_vector: list[float], limit: int = 5) -> list[VectorRecord]:
        scored = [
            (self._cosine_similarity(query_vector, record.vector), record)
            for record in self._records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def search_with_scores(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[tuple[VectorRecord, float]]:
        scored = [
            (record, self._cosine_similarity(query_vector, record.vector))
            for record in self._records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class QdrantVectorStore:
    def __init__(self, url: str, api_key: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "qdrant-client is required for QdrantVectorStore.") from exc

        self._client = QdrantClient(url=url, api_key=api_key)

    def upsert(self, collection: str, records: list[VectorRecord]) -> None:
        points = [
            {
                "id": record.record_id,
                "vector": record.vector,
                "payload": record.payload,
            }
            for record in records
        ]
        self._client.upsert(collection_name=collection, points=points)

    def search(self, collection: str, query_vector: list[float], limit: int = 5) -> list[dict]:
        return self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=limit,
        )
