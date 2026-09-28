from datetime import datetime, timezone

from app.citations.grounding import evaluate_grounding
from app.generation.context_builder import BuiltContext, ContextSource
from app.models.schemas import Chunk, CitationValidationResult, RetrievalCandidate


def _context_from_text(content: str):
    now = datetime.now(timezone.utc)
    chunk = Chunk(chunk_id="c1", document_id="d1", document_title="Policy Doc",
                  source_path="/docs/policy.md", document_type="md", department="hr",
                  section="Section", chunk_index=0, content=content, created_at=now, updated_at=now)
    cand = RetrievalCandidate(chunk=chunk, fusion_score=0.9)
    return BuiltContext(text="[Source 1]\n...", sources=[ContextSource(marker=1, candidate=cand)])


def test_grounded_answer_with_high_context_overlap_is_supported():
    context = _context_from_text("Employees must rotate their passwords every 90 days for security.")
    valid = CitationValidationResult(valid=True, citations_checked=1, invalid_citations=[])
    grounding = evaluate_grounding(
        "How often must employees rotate passwords?",
        "Employees must rotate their passwords every 90 days [1].",
        context, valid,
    )
    assert grounding.supported is True
    assert grounding.relevance_score > 0


def test_answer_with_low_context_overlap_flagged_unsupported():
    context = _context_from_text("Employees must rotate their passwords every 90 days for security.")
    valid = CitationValidationResult(valid=True, citations_checked=1, invalid_citations=[])
    grounding = evaluate_grounding(
        "How often must employees rotate passwords?",
        "The company offers unlimited vacation and free electric cars for everyone [1].",
        context, valid,
    )
    assert grounding.supported is False
    assert any("unsupported" in n.lower() for n in grounding.notes)


def test_empty_context_marks_insufficient_evidence():
    empty_context = BuiltContext(text="", sources=[])
    valid = CitationValidationResult(valid=True, citations_checked=0, invalid_citations=[])
    grounding = evaluate_grounding("Some question", "Some answer with content here.", empty_context, valid)
    assert grounding.has_sufficient_evidence is False
