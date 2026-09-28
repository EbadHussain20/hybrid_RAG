"""Optional reranking stage.

Architecture note: the reference spec asks for a cross-encoder reranker
(e.g. a small `cross-encoder/ms-marco-*` model). That requires the same
torch/transformers stack that is unavailable in this sandbox (no disk space,
no model-download network access - see app/embeddings/tfidf_svd.py for the
same constraint on the dense embedder).

`CrossEncoderReranker` below is implemented and ready to use wherever
sentence-transformers is installed (RERANKER_BACKEND=cross_encoder). The
default active reranker, `LexicalOverlapReranker`, is a genuine second-opinion
re-scorer that is NOT the same computation as either dense or sparse
retrieval: it scores query-chunk pairs using normalized token-overlap
(coverage of query terms + a length-normalized penalty for verbose chunks),
which behaves like a cheap relevance cross-check pass rather than duplicating
BM25 or cosine similarity. It is deliberately kept simple and fast so
reranking stays optional/toggleable without adding a hard dependency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import settings
from app.models.schemas import RetrievalCandidate
from app.retrieval.sparse import tokenize


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        ...


class LexicalOverlapReranker(BaseReranker):
    """Default, dependency-free reranker: query-term coverage with a length prior."""

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        query_terms = set(tokenize(query))
        if not query_terms:
            for i, c in enumerate(candidates[:top_k], start=1):
                c.rerank_score = c.fusion_score
                c.final_rank = i
            return candidates[:top_k]

        scored = []
        for cand in candidates:
            chunk_terms = tokenize(cand.chunk.content)
            chunk_term_set = set(chunk_terms)
            overlap = len(query_terms & chunk_term_set) / max(1, len(query_terms))
            # Mild penalty for very long chunks that pad in unrelated content.
            length_prior = 1.0 / (1.0 + len(chunk_terms) / 400.0)
            rerank_score = 0.7 * overlap + 0.2 * cand.fusion_score + 0.1 * length_prior
            cand.rerank_score = rerank_score
            scored.append(cand)

        scored.sort(key=lambda c: -c.rerank_score)
        for i, c in enumerate(scored[:top_k], start=1):
            c.final_rank = i
        return scored[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Real cross-encoder reranker - requires sentence-transformers + a downloaded model.

    Not active by default in this sandbox; see module docstring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        model = self._load()
        pairs = [(query, c.chunk.content) for c in candidates]
        scores = model.predict(pairs)
        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)
        candidates.sort(key=lambda c: -c.rerank_score)
        for i, c in enumerate(candidates[:top_k], start=1):
            c.final_rank = i
        return candidates[:top_k]


def get_reranker() -> BaseReranker:
    backend = getattr(settings, "RERANKER_BACKEND", "lexical_overlap")
    if backend == "cross_encoder":
        return CrossEncoderReranker()
    return LexicalOverlapReranker()
