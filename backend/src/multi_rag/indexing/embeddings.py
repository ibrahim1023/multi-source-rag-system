# Embedding providers.

from __future__ import annotations

import hashlib
from dataclasses import dataclass


class EmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class HashEmbeddingProvider(EmbeddingProvider):
    dim: int = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(self.dim):
            byte = digest[i % len(digest)]
            values.append(byte / 255.0)
        return values


@dataclass
class SentenceTransformerProvider(EmbeddingProvider):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "sentence-transformers is required for SentenceTransformerProvider."
            ) from exc

        model = SentenceTransformer(self.model_name)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
