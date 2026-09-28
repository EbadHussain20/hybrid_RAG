from datetime import datetime, timezone

from app.embeddings.tfidf_svd import TfidfSvdEmbeddingProvider
from app.models.schemas import Chunk
from app.retrieval.dense import DenseRetriever


def _make_chunk(chunk_id, doc_id, content):
    now = datetime.now(timezone.utc)
    return Chunk(chunk_id=chunk_id, document_id=doc_id, document_title="Doc",
                 source_path=f"/docs/{doc_id}.md", document_type="md", department="hr",
                 section=None, chunk_index=0, content=content, created_at=now, updated_at=now)


def test_dense_retriever_returns_normalized_scores_in_range():
    chunks = [
        _make_chunk("c1", "d1", "Employees must rotate their passwords every ninety days for security."),
        _make_chunk("c2", "d2", "The kitchen restocks snacks every Monday morning."),
        _make_chunk("c3", "d3", "Password rotation is a core part of the security policy."),
    ]
    retriever = DenseRetriever(provider=TfidfSvdEmbeddingProvider(dim=8))
    retriever.build(chunks)
    results = retriever.search("password security rotation", top_k=3)
    assert len(results) == 3
    for _, score, rank in results:
        assert -1.0001 <= score <= 1.0001

    # The two password-related chunks should outrank the unrelated kitchen chunk.
    ranked_ids = [c.chunk_id for c, _, _ in results]
    assert ranked_ids.index("c2") == 2  # kitchen chunk ranks last


def test_dense_retriever_empty_index_returns_empty():
    retriever = DenseRetriever(provider=TfidfSvdEmbeddingProvider(dim=8))
    assert retriever.search("anything", top_k=5) == []


def test_embedding_provider_persistence_roundtrip(tmp_path):
    provider = TfidfSvdEmbeddingProvider(dim=8)
    provider.fit(["password policy security", "vacation approval manager", "remote work equipment"])
    vec_before = provider.encode(["password policy"])

    save_path = tmp_path / "provider.pkl"
    provider.save(str(save_path))

    reloaded = TfidfSvdEmbeddingProvider(dim=8)
    reloaded.load(str(save_path))
    vec_after = reloaded.encode(["password policy"])

    assert (vec_before == vec_after).all()
