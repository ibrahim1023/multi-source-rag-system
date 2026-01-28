# LLM-based answerer that synthesizes a response from grounded claims.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from multi_rag.answering.grounded import GroundedAnswerer
from multi_rag.models import AnswerResponse, Claim, Chunk
from multi_rag.retrieval.hybrid import RetrievalResult


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class LLMAnsweringConfig:
    model_name: str
    refusal_text: str = "I do not have enough context to answer that yet."


class GeminiClient:
    def __init__(self, *, api_key: str, model_name: str) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "google-genai is required for Gemini answering."
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._ensure_client()
        response = client.models.generate_content(model=self._model_name, contents=prompt)
        text = getattr(response, "text", "") or ""
        return text.strip()


class LLMAnswerer:
    def __init__(
        self,
        *,
        client: LLMClient,
        grounded: GroundedAnswerer,
        config: LLMAnsweringConfig,
    ) -> None:
        self._client = client
        self._grounded = grounded
        self._config = config

    def answer(
        self,
        query: str,
        results: list[RetrievalResult],
        context: list[Chunk],
    ) -> AnswerResponse:
        grounded_response = self._grounded.answer(query, results, context)
        if grounded_response.refused:
            return grounded_response
        if not grounded_response.claims:
            return grounded_response
        prompt = _build_prompt(query, grounded_response.claims, self._config.refusal_text)
        llm_answer = self._client.generate(prompt).strip()
        if not llm_answer:
            return grounded_response
        grounded_response.answer = llm_answer
        return grounded_response


def _build_prompt(query: str, claims: list[Claim], refusal_text: str) -> str:
    claim_lines = "\n".join(f"- {claim.text}" for claim in claims if claim.text)
    return (
        "You are a helpful assistant answering questions using ONLY the facts below.\n"
        "If the facts are insufficient, respond with this exact sentence:\n"
        f"{refusal_text}\n\n"
        f"Question: {query}\n"
        "Facts:\n"
        f"{claim_lines}\n\n"
        "Answer in 2-5 sentences. Do not add citations."
    )
