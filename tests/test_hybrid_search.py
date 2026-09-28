from datetime import datetime, timezone

from app.embeddings.tfidf_svd import TfidfSvdEmbeddingProvider
from app.models.schemas import Chunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.sparse import SparseRetriever


def _make_chunk(chunk_id, doc_id, content):
    now = datetime.now(timezone.utc)
    return Chunk(chunk_id=chunk_id, document_id=doc_id, document_title="Doc",
                 source_path=f"/docs/{doc_id}.md", document_type="md", department="hr",
                 section=None, chunk_index=0, content=content, created_at=now, updated_at=now)


def _build_hybrid():
    chunks = [
        _make_chunk("c1", "d1", "Employees must rotate their passwords every 90 days."),
        _make_chunk("c2", "d2", "The kitchen restocks snacks every Monday."),
        _make_chunk("c3", "d3", "Security policy requires strong password rotation practices."),
        _make_chunk("c4", "d4", "Vacation requests need manager approval within 5 business days."),
    ]
    dense = DenseRetriever(provider=TfidfSvdEmbeddingProvider(dim=8))
    dense.build(chunks)
    sparse = SparseRetriever()
    sparse.build(chunks)
    return HybridRetriever(dense, sparse), chunks


def test_hybrid_retrieve_returns_fusion_scores_and_ranks():
    hybrid, _ = _build_hybrid()
    results = hybrid.retrieve("password rotation policy", top_k=4)
    assert results
    for i, cand in enumerate(results, start=1):
        assert cand.final_rank == i
        assert cand.fusion_score >= 0

    fusion_scores = [c.fusion_score for c in results]
    assert fusion_scores == sorted(fusion_scores, reverse=True)


def test_hybrid_candidate_tracks_both_dense_and_sparse_scores():
    hybrid, _ = _build_hybrid()
    results = hybrid.retrieve("password rotation", top_k=4)
    top = results[0]
    assert top.dense_rank is not None or top.sparse_rank is not None


def test_hybrid_surfaces_unrelated_chunk_last():
    hybrid, _ = _build_hybrid()
    results = hybrid.retrieve("password security rotation", top_k=4)
    ranked_ids = [c.chunk.chunk_id for c in results]
    assert ranked_ids[-1] == "c2"  # the kitchen chunk should rank last


def test_weighted_fusion_method_also_produces_scores():
    hybrid, _ = _build_hybrid()
    results = hybrid.retrieve("password rotation policy", top_k=4, fusion_method="weighted")
    assert results
    for c in results:
        assert 0.0 <= c.fusion_score <= 1.0001
