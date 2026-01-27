# Pydantic models for the API layer.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from multi_rag.models import AnswerResponse, Claim, Citation


class IngestRequest(BaseModel):
    title: str
    origin: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    metadata_filter: dict[str, Any] | None = None


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    chunk_text: str
    metadata: dict[str, Any]
    section_path: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ChatRequest(BaseModel):
    query: str
    metadata_filter: dict[str, Any] | None = None


class CitationModel(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    origin: str
    section_path: str | None = None
    snippet: str

    @classmethod
    def from_citation(cls, citation: Citation) -> "CitationModel":
        return cls(
            doc_id=citation.doc_id,
            chunk_id=citation.chunk_id,
            title=citation.title,
            origin=citation.origin,
            section_path=citation.section_path,
            snippet=citation.snippet,
        )


class ClaimModel(BaseModel):
    text: str
    citations: list[CitationModel]

    @classmethod
    def from_claim(cls, claim: Claim) -> "ClaimModel":
        return cls(
            text=claim.text,
            citations=[CitationModel.from_citation(item) for item in claim.citations],
        )


class ChatResponse(BaseModel):
    answer: str
    claims: list[ClaimModel]
    citations: list[CitationModel]
    confidence: float
    mode: str
    follow_up_question: str | None = None
    refused: bool

    @classmethod
    def from_response(cls, response: AnswerResponse) -> "ChatResponse":
        return cls(
            answer=response.answer,
            claims=[ClaimModel.from_claim(item) for item in response.claims],
            citations=[CitationModel.from_citation(item) for item in response.citations],
            confidence=response.confidence,
            mode=response.mode,
            follow_up_question=response.follow_up_question,
            refused=response.refused,
        )
