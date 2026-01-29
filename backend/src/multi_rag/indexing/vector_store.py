# Vector store implementations.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol
import uuid


@dataclass
class VectorRecord:
    record_id: str
    vector: list[float]
    payload: dict


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None:
        raise NotImplementedError

    def delete_by_ids(self, record_ids: set[str]) -> None:
        raise NotImplementedError

    def search_with_scores(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[tuple[VectorRecord, float]]:
        raise NotImplementedError


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: list[VectorRecord] = []

    def upsert(self, records: list[VectorRecord]) -> None:
        self._records.extend(records)

    def delete_by_ids(self, record_ids: set[str]) -> None:
        if not record_ids:
            return
        self._records = [
            record for record in self._records if record.record_id not in record_ids
        ]

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
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        collection: str = "multi_rag",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "qdrant-client is required for QdrantVectorStore.") from exc

        self._client = QdrantClient(url=url, api_key=api_key)
        self._models = qmodels
        self._collection = collection
        self._collection_ready = False

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._ensure_collection(len(records[0].vector))
        points = [
            {
                "id": self._point_id(record.record_id),
                "vector": record.vector,
                "payload": self._payload(record),
            }
            for record in records
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def delete_by_ids(self, record_ids: set[str]) -> None:
        if not record_ids:
            return
        if not self._ensure_collection():
            return
        selector = self._models.PointIdsList(
            points=[self._point_id(record_id) for record_id in record_ids]
        )
        self._client.delete(
            collection_name=self._collection,
            points_selector=selector,
        )

    def search_with_scores(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[tuple[VectorRecord, float]]:
        if not self._ensure_collection():
            return []
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        scored: list[tuple[VectorRecord, float]] = []
        for point in results:
            payload = point.payload if point.payload is not None else {}
            scored.append(
                (
                    VectorRecord(
                        record_id=str(payload.get("chunk_id", point.id)),
                        vector=[],
                        payload=payload,
                    ),
                    float(point.score),
                )
            )
        return scored

    def _ensure_collection(self, vector_size: int | None = None) -> bool:
        if self._collection_ready:
            return True
        try:
            self._client.get_collection(self._collection)
        except Exception:
            if vector_size is None:
                return False
            params = self._models.VectorParams(
                size=vector_size, distance=self._models.Distance.COSINE
            )
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=params,
            )
        self._collection_ready = True
        return True

    @staticmethod
    def _point_id(record_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))

    @staticmethod
    def _payload(record: VectorRecord) -> dict:
        payload = dict(record.payload)
        payload.setdefault("chunk_id", record.record_id)
        return payload
