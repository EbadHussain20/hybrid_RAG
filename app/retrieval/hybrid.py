"""Hybrid retrieval: independently query dense + sparse, then fuse.

Two fusion methods are supported (FUSION_METHOD in config):

1. Reciprocal Rank Fusion (RRF) - the default. For each candidate,
   fusion_score = sum over retrievers r of 1 / (RRF_K + rank_r).
   RRF is rank-based rather than score-based, which sidesteps the problem
   that BM25 scores and cosine similarities live on incomparable scales -
   a well documented issue with naively summing raw scores. RRF_K (default 60)
   dampens the influence of very top ranks so one retriever can't dominate
   purely because of scale.

2. Weighted score fusion - min-max normalizes dense and sparse scores to
   [0, 1] independently, then combines as
   fusion_score = DENSE_WEIGHT * norm(dense_score) + SPARSE_WEIGHT * norm(sparse_score).
   This is more interpretable/tunable but more sensitive to score distribution
   quirks, which is why RRF is the default.

A chunk retrieved by only one retriever still gets a fusion score (its missing
side simply contributes 0), so hybrid retrieval can surface pure-keyword or
pure-semantic matches that the other retriever missed entirely.
"""
from __future__ import annotations

from app.core.config import settings
from app.models.schemas import Chunk, RetrievalCandidate
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse

    def retrieve(self, query: str, top_k: int | None = None,
                 dense_top_k: int | None = None, sparse_top_k: int | None = None,
                 fusion_method: str | None = None) -> list[RetrievalCandidate]:
        top_k = top_k or settings.HYBRID_TOP_K
        dense_top_k = dense_top_k or settings.DENSE_TOP_K
        sparse_top_k = sparse_top_k or settings.SPARSE_TOP_K
        fusion_method = fusion_method or settings.FUSION_METHOD

        dense_results = self.dense.search(query, top_k=dense_top_k)
        sparse_results = self.sparse.search(query, top_k=sparse_top_k)

        by_chunk_id: dict[str, RetrievalCandidate] = {}

        for chunk, score, rank in dense_results:
            by_chunk_id[chunk.chunk_id] = RetrievalCandidate(
                chunk=chunk, dense_score=score, dense_rank=rank,
            )

        for chunk, score, rank, matched in sparse_results:
            if chunk.chunk_id in by_chunk_id:
                cand = by_chunk_id[chunk.chunk_id]
                cand.sparse_score = score
                cand.sparse_rank = rank
                cand.matched_terms = matched
            else:
                by_chunk_id[chunk.chunk_id] = RetrievalCandidate(
                    chunk=chunk, sparse_score=score, sparse_rank=rank, matched_terms=matched,
                )

        candidates = list(by_chunk_id.values())

        if fusion_method == "weighted":
            dense_scores = _min_max_normalize([c.dense_score for c in candidates])
            sparse_scores = _min_max_normalize([c.sparse_score for c in candidates])
            for cand, d_norm, s_norm in zip(candidates, dense_scores, sparse_scores):
                cand.fusion_score = settings.DENSE_WEIGHT * d_norm + settings.SPARSE_WEIGHT * s_norm
        else:  # rrf (default)
            k = settings.RRF_K
            for cand in candidates:
                score = 0.0
                if cand.dense_rank is not None:
                    score += 1.0 / (k + cand.dense_rank)
                if cand.sparse_rank is not None:
                    score += 1.0 / (k + cand.sparse_rank)
                cand.fusion_score = score

        candidates.sort(key=lambda c: -c.fusion_score)

        # Deduplicate (defensive - chunk_ids are already unique keys, but chunks
        # from the same document/section with near-identical content are capped
        # for diversity in context_builder, not here).
        for i, cand in enumerate(candidates[:top_k], start=1):
            cand.final_rank = i

        return candidates[:top_k]
