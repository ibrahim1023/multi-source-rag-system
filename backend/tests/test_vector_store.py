# Tests for vector store selection and Qdrant adapter.

from __future__ import annotations

import sys
import types


def _install_fake_qdrant(monkeypatch):
    class FakePoint:
        def __init__(self, point_id, payload, score):
            self.id = point_id
            self.payload = payload
            self.score = score

    class FakeQdrantClient:
        def __init__(self, url, api_key=None):
            self.url = url
            self.api_key = api_key
            self.collections = {}

        def get_collection(self, collection_name):
            if collection_name not in self.collections:
                raise RuntimeError("missing")
            return self.collections[collection_name]

        def create_collection(self, collection_name, vectors_config):
            self.collections[collection_name] = {
                "vectors_config": vectors_config,
                "points": {},
            }

        def upsert(self, collection_name, points):
            self.collections.setdefault(collection_name, {"points": {}})
            storage = self.collections[collection_name]["points"]
            for point in points:
                storage[str(point["id"])] = point

        def search(self, collection_name, query_vector, limit=5, with_payload=True):
            collection = self.collections.get(collection_name)
            if not collection:
                return []
            points = list(collection["points"].values())[:limit]
            return [
                FakePoint(point["id"], point.get("payload", {}), 1.0)
                for point in points
            ]

        def delete(self, collection_name, points_selector):
            collection = self.collections.get(collection_name)
            if not collection:
                return
            storage = collection["points"]
            for point_id in points_selector.points:
                storage.pop(str(point_id), None)

    class FakeModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        class PointIdsList:
            def __init__(self, points):
                self.points = points

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.QdrantClient = FakeQdrantClient

    fake_http = types.ModuleType("qdrant_client.http")
    fake_models = types.ModuleType("qdrant_client.http.models")
    fake_models.Distance = FakeModels.Distance
    fake_models.VectorParams = FakeModels.VectorParams
    fake_models.PointIdsList = FakeModels.PointIdsList
    fake_http.models = fake_models

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", fake_http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", fake_models)


def test_qdrant_vector_store_round_trip(monkeypatch):
    _install_fake_qdrant(monkeypatch)

    from multi_rag.indexing.vector_store import QdrantVectorStore, VectorRecord
    import uuid

    store = QdrantVectorStore(url="http://fake", api_key="key", collection="test")
    records = [
        VectorRecord(record_id="a", vector=[0.1, 0.2], payload={"doc_id": "1"}),
        VectorRecord(record_id="b", vector=[0.2, 0.3], payload={"doc_id": "1"}),
    ]
    store.upsert(records)

    points = list(store._client.collections["test"]["points"].values())
    ids = [point["id"] for point in points]
    assert ids == [
        str(uuid.uuid5(uuid.NAMESPACE_URL, "a")),
        str(uuid.uuid5(uuid.NAMESPACE_URL, "b")),
    ]

    results = store.search_with_scores([0.1, 0.2], limit=2)
    ids = [record.record_id for record, _ in results]
    assert ids == ["a", "b"]

    store.delete_by_ids({"a"})
    results = store.search_with_scores([0.1, 0.2], limit=2)
    ids = [record.record_id for record, _ in results]
    assert ids == ["b"]


def test_selects_qdrant_when_configured(monkeypatch):
    _install_fake_qdrant(monkeypatch)

    from multi_rag.api.app import APISettings, build_dependencies
    from multi_rag.indexing.vector_store import QdrantVectorStore

    settings = APISettings(qdrant_url="http://fake", qdrant_collection="test")
    deps = build_dependencies(settings)
    assert isinstance(deps.indexer._stores.vector_store, QdrantVectorStore)
