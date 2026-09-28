from datetime import datetime, timezone

from app.models.schemas import Chunk
from app.retrieval.sparse import SparseRetriever, tokenize


def _make_chunk(chunk_id, doc_id, content, title="Doc"):
    now = datetime.now(timezone.utc)
    return Chunk(chunk_id=chunk_id, document_id=doc_id, document_title=title,
                 source_path=f"/docs/{doc_id}.md", document_type="md", department="hr",
                 section=None, chunk_index=0, content=content, created_at=now, updated_at=now)


def test_tokenize_lowercases_and_splits():
    assert tokenize("Password Rotation-90Days!") == ["password", "rotation", "90days"]


def test_bm25_retrieves_relevant_chunk_first():
    chunks = [
        _make_chunk("c1", "d1", "Employees must rotate passwords every 90 days."),
        _make_chunk("c2", "d2", "The office has free coffee and snacks."),
        _make_chunk("c3", "d3", "Vacation requests need manager approval."),
    ]
    retriever = SparseRetriever()
    retriever.build(chunks)
    # Note: BM25 here does exact-token keyword matching with no stemming (by design -
    # this is what distinguishes it from the semantic dense retriever), so the query
    # must share literal tokens with the target chunk.
    results = retriever.search("rotate passwords days", top_k=3)
    assert results[0][0].chunk_id == "c1"


def test_bm25_returns_matched_terms():
    # rank_bm25's IDF is degenerate (=0) for a term appearing in exactly half of a
    # 2-document corpus, so use a slightly larger corpus for a realistic IDF spread.
    chunks = [
        _make_chunk("c1", "d1", "Remote work requires manager approval."),
        _make_chunk("c2", "d2", "The kitchen restocks snacks every Monday."),
        _make_chunk("c3", "d3", "Quarterly budget review happens every October."),
    ]
    retriever = SparseRetriever()
    retriever.build(chunks)
    results = retriever.search("manager approval work", top_k=3)
    assert results
    _, score, rank, matched = results[0]
    assert score > 0
    assert "manager" in matched and "approval" in matched


def test_bm25_no_results_for_empty_index():
    retriever = SparseRetriever()
    retriever.build([])
    assert retriever.search("anything", top_k=5) == []


def test_bm25_excludes_zero_score_chunks():
    chunks = [
        _make_chunk("c1", "d1", "Completely unrelated content about coffee machines."),
    ]
    retriever = SparseRetriever()
    retriever.build(chunks)
    results = retriever.search("xyzxyz_not_present_anywhere", top_k=5)
    assert results == []
