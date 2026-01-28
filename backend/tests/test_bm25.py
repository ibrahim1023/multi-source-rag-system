# Unit tests for BM25 index.

from __future__ import annotations

from multi_rag.indexing.bm25 import BM25Index


def test_bm25_ranks_relevant_document_higher() -> None:
    index = BM25Index()
    index.add_documents(
        ["doc1", "doc2"],
        ["retention policy for logs", "incident response checklist"],
    )
    results = index.search("retention logs", top_k=2)
    assert results[0][0] == "doc1"


def test_bm25_persists_and_loads(tmp_path) -> None:
    path = tmp_path / "bm25.json"
    index = BM25Index(persist_path=str(path))
    index.add_documents(
        ["doc1", "doc2"],
        ["retention policy for logs", "incident response checklist"],
    )

    loaded = BM25Index()
    loaded.load(str(path))
    results = loaded.search("retention logs", top_k=1)
    assert results[0][0] == "doc1"


def test_bm25_deletes_documents() -> None:
    index = BM25Index()
    index.add_documents(
        ["doc1", "doc2", "doc3"],
        ["retention policy for logs", "incident response checklist", "retention notes"],
    )

    index.delete_documents(["doc1"])
    results = index.search("retention logs", top_k=2)
    assert all(doc_id != "doc1" for doc_id, _ in results)
