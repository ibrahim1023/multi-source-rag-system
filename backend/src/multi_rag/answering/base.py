# Answerer protocol for pipeline wiring.

from __future__ import annotations

from typing import Protocol

from multi_rag.models import AnswerResponse, Chunk
from multi_rag.retrieval.hybrid import RetrievalResult


class Answerer(Protocol):
    def answer(
        self,
        query: str,
        results: list[RetrievalResult],
        context: list[Chunk],
    ) -> AnswerResponse: ...
