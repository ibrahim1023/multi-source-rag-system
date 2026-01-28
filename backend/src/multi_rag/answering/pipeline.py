# Retrieval-to-answer orchestration.

from __future__ import annotations

from dataclasses import dataclass

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.models import AnswerResponse
from multi_rag.observability.tracing import NullTracer, Tracer
from multi_rag.retrieval.hybrid import HybridRetriever


@dataclass
class AnsweringPipelineConfig:
    top_k: int = 5
    context_max_chunks: int = 6
    neighbor_window: int = 1
    focus_top_doc: bool = True


class AnsweringPipeline:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        answerer: GroundedAnswerer,
        config: AnsweringPipelineConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._config = config or AnsweringPipelineConfig()
        self._tracer = tracer or NullTracer()

    def answer(self, query: str, *, metadata_filter: dict | None = None) -> AnswerResponse:
        results = self._retriever.retrieve(
            query,
            top_k=self._config.top_k,
            metadata_filter=metadata_filter,
        )
        self._tracer.record_event(
            "retrieval.completed",
            {
                "query": query,
                "top_k": self._config.top_k,
                "result_count": len(results),
            },
        )
        context = self._retriever.assemble_context(
            results,
            max_chunks=self._config.context_max_chunks,
            neighbor_window=self._config.neighbor_window,
            focus_top_doc=self._config.focus_top_doc,
        )
        response = self._answerer.answer(query, results, context)
        if metadata_filter and response.follow_up_question:
            response.follow_up_question = None
        self._tracer.record_event(
            "answer.completed",
            {
                "query": query,
                "mode": response.mode,
                "confidence": response.confidence,
                "claim_count": len(response.claims),
                "citation_count": len(response.citations),
            },
        )
        return response
