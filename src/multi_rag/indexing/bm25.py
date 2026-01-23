# Simple BM25 index for keyword search.

from __future__ import annotations

from dataclasses import dataclass
import math
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    def __init__(self, config: BM25Config | None = None) -> None:
        self._config = config or BM25Config()
        self._docs: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len = 0.0

    def add_documents(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must be the same length.")

        for doc_id, text in zip(doc_ids, texts):
            tokens = _tokenize(text)
            self._doc_ids.append(doc_id)
            self._docs.append(tokens)
            seen = set(tokens)
            for token in seen:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        self._avg_doc_len = sum(len(doc)
                                for doc in self._docs) / max(len(self._docs), 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        scores: list[tuple[str, float]] = []
        for doc_id, doc_tokens in zip(self._doc_ids, self._docs):
            score = self._score(tokens, doc_tokens)
            scores.append((doc_id, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        score = 0.0
        doc_len = len(doc_tokens)
        freqs: dict[str, int] = {}
        for token in doc_tokens:
            freqs[token] = freqs.get(token, 0) + 1

        for token in query_tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self._docs) - df + 0.5) / (df + 0.5))
            tf = freqs.get(token, 0)
            denom = tf + self._config.k1 * \
                (1 - self._config.b + self._config.b * doc_len / self._avg_doc_len)
            score += idf * (tf * (self._config.k1 + 1)) / denom
        return score
