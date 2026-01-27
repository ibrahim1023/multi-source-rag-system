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
