# Unit tests for LLM-backed answering.

from __future__ import annotations

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.answering.llm import LLMAnswerer, LLMAnsweringConfig
from multi_rag.indexing.metadata_store import InMemoryMetadataStore
from multi_rag.models import Chunk, Document
from multi_rag.retrieval.hybrid import RetrievalResult


class FakeLLMClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


def test_llm_answerer_uses_llm_answer_and_keeps_citations() -> None:
    metadata_store = InMemoryMetadataStore()
    document = Document(
        doc_id="doc1",
        source_type="markdown",
        title="Policy",
        origin="/tmp/policy.md",
    )
    metadata_store.upsert_document(document)

    chunk = Chunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        chunk_text="Retention policy for logs is 30 days. Keep backups monthly.",
        chunk_index=1,
        section_path="Retention",
        metadata={"source_type": "markdown"},
    )
    metadata_store.upsert_chunk(chunk)

    grounded = GroundedAnswerer(metadata_store=metadata_store)
    client = FakeLLMClient("Synthesized answer from Gemini.")
    answerer = LLMAnswerer(
        client=client,
        grounded=grounded,
        config=LLMAnsweringConfig(model_name="gemini-flash-latest"),
    )

    results = [RetrievalResult(chunk_id="doc1#0001", score=0.9, payload={})]
    response = answerer.answer("retention logs", results, [chunk])

    assert response.refused is False
    assert response.answer == "Synthesized answer from Gemini."
    assert response.citations
    assert response.citations[0].chunk_id == "doc1#0001"
    assert client.prompts


def test_llm_answerer_refuses_without_context() -> None:
    metadata_store = InMemoryMetadataStore()
    grounded = GroundedAnswerer(metadata_store=metadata_store)
    client = FakeLLMClient("Should not be used.")
    answerer = LLMAnswerer(
        client=client,
        grounded=grounded,
        config=LLMAnsweringConfig(model_name="gemini-flash-latest"),
    )

    response = answerer.answer("retention logs", [], [])

    assert response.refused is True
    assert not client.prompts
