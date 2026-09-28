"""Sparse keyword retriever using genuine BM25 (Okapi BM25 via rank_bm25).

This operates entirely independently of the dense index: its own tokenization,
its own scoring (term frequency / inverse document frequency with length
normalization), and it can be inspected directly (score + matching terms) as
required by the spec.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def load(self, chunks: list[Chunk]) -> None:
        self.build(chunks)  # BM25 index is cheap to rebuild from chunk text at load time

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float, int, list[str]]]:
        """Returns list of (chunk, bm25_score, rank[1-indexed], matched_terms)."""
        if self.bm25 is None or not self.chunks:
            return []
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        results = []
        query_token_set = set(query_tokens)
        for rank, idx in enumerate(ranked_idx, start=1):
            if scores[idx] <= 0:
                continue
            matched = sorted(query_token_set & set(self.tokenized_corpus[idx]))
            results.append((self.chunks[idx], float(scores[idx]), rank, matched))
        return results
