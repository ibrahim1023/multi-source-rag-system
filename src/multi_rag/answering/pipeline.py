# Retrieval-to-answer orchestration.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.models import AnswerResponse
from multi_rag.retrieval.hybrid import HybridRetriever


@dataclass
class AnsweringPipelineConfig:
    top_k: int = 5
    context_max_chunks: int = 6
    neighbor_window: int = 1


class AnsweringPipeline:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        answerer: GroundedAnswerer,
        config: AnsweringPipelineConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._config = config or AnsweringPipelineConfig()

    def answer(self, query: str, *, metadata_filter: dict | None = None) -> AnswerResponse:
        results = self._retriever.retrieve(
            query,
            top_k=self._config.top_k,
            metadata_filter=metadata_filter,
        )
        context = self._retriever.assemble_context(
            results,
            max_chunks=self._config.context_max_chunks,
            neighbor_window=self._config.neighbor_window,
        )
        return self._answerer.answer(query, results, context)
