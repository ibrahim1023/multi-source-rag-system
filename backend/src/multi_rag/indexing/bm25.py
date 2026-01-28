# Simple BM25 index for keyword search.

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    def __init__(
        self,
        config: BM25Config | None = None,
        *,
        persist_path: str | None = None,
    ) -> None:
        self._config = config or BM25Config()
        self._docs: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._doc_freq: dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._persist_path = persist_path

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
        self.persist()

    def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        remove_ids = set(doc_ids)
        remaining_docs: list[list[str]] = []
        remaining_ids: list[str] = []
        for doc_id, doc_tokens in zip(self._doc_ids, self._docs):
            if doc_id in remove_ids:
                continue
            remaining_ids.append(doc_id)
            remaining_docs.append(doc_tokens)
        self._doc_ids = remaining_ids
        self._docs = remaining_docs
        self._doc_freq = {}
        for doc_tokens in self._docs:
            for token in set(doc_tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avg_doc_len = sum(len(doc)
                                for doc in self._docs) / max(len(self._docs), 1)
        self.persist()

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

    def persist(self, path: str | None = None) -> None:
        target = path or self._persist_path
        if not target:
            return
        payload = {
            "version": 1,
            "config": {"k1": self._config.k1, "b": self._config.b},
            "doc_ids": self._doc_ids,
            "docs": self._docs,
            "doc_freq": self._doc_freq,
            "avg_doc_len": self._avg_doc_len,
        }
        Path(target).write_text(json.dumps(payload))

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        if data.get("version") != 1:
            raise ValueError("Unsupported BM25 persistence format.")
        config = data.get("config") or {}
        self._config = BM25Config(
            k1=float(config.get("k1", 1.5)),
            b=float(config.get("b", 0.75)),
        )
        self._doc_ids = list(data.get("doc_ids", []))
        self._docs = list(data.get("docs", []))
        self._doc_freq = {str(k): int(v) for k, v in data.get("doc_freq", {}).items()}
        self._avg_doc_len = float(data.get("avg_doc_len", 0.0))
