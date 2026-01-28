# Query expansion and stopword handling.

from __future__ import annotations

from dataclasses import dataclass, field
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_DEFAULT_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "s",
    "she",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class QueryExpansionConfig:
    enabled: bool = True
    max_total_terms: int = 24
    stopwords: set[str] = field(default_factory=lambda: set(_DEFAULT_STOPWORDS))
    use_expanded_for_embeddings: bool = True
    use_expanded_for_bm25: bool = True


@dataclass(frozen=True)
class ExpandedQuery:
    original: str
    keywords: list[str]
    expanded_tokens: list[str]
    expanded_text: str


def expand_query(query: str, config: QueryExpansionConfig | None = None) -> ExpandedQuery:
    active = config or QueryExpansionConfig()
    tokens = [token.lower() for token in _TOKEN_RE.findall(query)]
    keywords = [token for token in tokens if token not in active.stopwords]
    if not keywords:
        keywords = tokens
    expanded_tokens = list(keywords)
    if active.enabled:
        expanded_tokens = _expand_terms(keywords, active.max_total_terms)
    expanded_text = " ".join(expanded_tokens) if expanded_tokens else query
    return ExpandedQuery(
        original=query,
        keywords=keywords,
        expanded_tokens=expanded_tokens,
        expanded_text=expanded_text,
    )


def _expand_terms(tokens: list[str], max_total_terms: int) -> list[str]:
    seen: set[str] = set()
    expanded: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        expanded.append(token)
        for variant in _variants(token):
            if len(expanded) >= max_total_terms:
                return expanded
            if variant in seen:
                continue
            seen.add(variant)
            expanded.append(variant)
    return expanded


def _variants(token: str) -> list[str]:
    variants: list[str] = []
    if len(token) <= 2:
        return variants
    if token.endswith("ies") and len(token) > 3:
        variants.append(f"{token[:-3]}y")
    if token.endswith("s") and len(token) > 3 and not token.endswith("ies"):
        variants.append(token[:-1])
    if token.endswith("y") and len(token) > 3:
        variants.append(f"{token[:-1]}ies")
    if not token.endswith("s"):
        variants.append(f"{token}s")
    return variants
