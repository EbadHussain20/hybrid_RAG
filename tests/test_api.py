import pytest
from fastapi.testclient import TestClient

from app.api.main import app, _state
from app.embeddings.tfidf_svd import TfidfSvdEmbeddingProvider
from app.models.schemas import Chunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.core.rag_pipeline import RagPipeline
from datetime import datetime, timezone


def _make_chunk(chunk_id, doc_id, content, title="Doc"):
    now = datetime.now(timezone.utc)
    return Chunk(chunk_id=chunk_id, document_id=doc_id, document_title=title,
                 source_path=f"/docs/{doc_id}.md", document_type="md", department="hr",
                 section="Section", chunk_index=0, content=content, created_at=now, updated_at=now)


@pytest.fixture
def client_with_index():
    chunks = [
        _make_chunk("c1", "d1", "Employees must rotate their passwords every 90 days.", "Security Policy"),
        _make_chunk("c2", "d2", "Vacation requests need manager approval.", "Employee Handbook"),
    ]
    dense = DenseRetriever(provider=TfidfSvdEmbeddingProvider(dim=8))
    dense.build(chunks)
    sparse = SparseRetriever()
    sparse.build(chunks)

    _state["dense"] = dense
    _state["sparse"] = sparse
    _state["pipeline"] = RagPipeline(dense, sparse)

    with TestClient(app) as c:
        yield c

    _state["pipeline"] = None
    _state["dense"] = None
    _state["sparse"] = None


def test_health_endpoint(client_with_index):
    resp = client_with_index.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["index_loaded"] is True


def test_metrics_endpoint(client_with_index):
    resp = client_with_index.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["num_indexed_chunks"] == 2


def test_query_endpoint_returns_grounded_answer(client_with_index):
    resp = client_with_index.post("/query", json={"question": "How often must passwords be rotated?", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert body["citation_validation"]["citations_checked"] >= 0
    assert "retrieval" in body


def test_query_endpoint_rejects_empty_question(client_with_index):
    resp = client_with_index.post("/query", json={"question": "   "})
    assert resp.status_code == 422


def test_search_endpoint_hybrid_mode(client_with_index):
    resp = client_with_index.post("/search", json={"query": "password rotation", "top_k": 2, "mode": "hybrid"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "hybrid"
    assert len(body["results"]) <= 2


def test_search_endpoint_dense_mode(client_with_index):
    resp = client_with_index.post("/search", json={"query": "password rotation", "top_k": 2, "mode": "dense"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "dense"


def test_query_endpoint_without_index_returns_503():
    _state["pipeline"] = None
    _state["dense"] = DenseRetriever(provider=TfidfSvdEmbeddingProvider(dim=8))
    _state["sparse"] = SparseRetriever()
    _state["pipeline"] = RagPipeline(_state["dense"], _state["sparse"])
    with TestClient(app) as c:
        resp = c.post("/query", json={"question": "anything?"})
    assert resp.status_code == 503
