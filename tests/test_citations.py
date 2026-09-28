from app.citations.grounding import evaluate_grounding
from app.citations.validator import build_citations, extract_citation_markers, validate_citations
from app.generation.context_builder import ContextSource, BuiltContext
from app.models.schemas import Chunk, RetrievalCandidate
from datetime import datetime, timezone


def _make_candidate(chunk_id, content, title="Employee Security Policy", section="Password Requirements"):
    now = datetime.now(timezone.utc)
    chunk = Chunk(chunk_id=chunk_id, document_id="doc1", document_title=title,
                  source_path="/docs/security/policy.md", document_type="md", department="security",
                  section=section, chunk_index=0, content=content, created_at=now, updated_at=now)
    return RetrievalCandidate(chunk=chunk, dense_score=0.8, sparse_score=5.0, fusion_score=0.9)


def _context_with_one_source():
    cand = _make_candidate("c1", "Employees must rotate their passwords every 90 days.")
    return BuiltContext(text="[Source 1]\n...", sources=[ContextSource(marker=1, candidate=cand)])


def test_extract_citation_markers_ignores_sources_section():
    answer = "Passwords rotate every 90 days [1].\nSources:\n[1] Policy — Section\n[2] Should not count"
    markers = extract_citation_markers(answer)
    assert markers == [1]


def test_extract_citation_markers_deduplicates_and_preserves_order():
    answer = "Fact A [2]. Fact B [1]. Fact C [2]."
    assert extract_citation_markers(answer) == [2, 1]


def test_validate_citations_accepts_real_marker():
    context = _context_with_one_source()
    answer = "Passwords must be rotated every 90 days [1]."
    result = validate_citations(answer, context)
    assert result.valid is True
    assert result.citations_checked == 1
    assert result.invalid_citations == []


def test_validate_citations_flags_hallucinated_marker():
    context = _context_with_one_source()
    answer = "Passwords must be rotated every 90 days [1]. Also see the fabricated policy [7]."
    result = validate_citations(answer, context)
    assert result.valid is False
    assert 7 in result.invalid_citations


def test_validate_citations_flags_missing_citation_on_non_noanswer_response():
    context = _context_with_one_source()
    answer = "Passwords must be rotated every 90 days."  # no citation at all
    result = validate_citations(answer, context)
    assert result.citations_checked == 0
    assert result.details  # should note the missing citation


def test_build_citations_maps_marker_to_real_chunk_metadata():
    context = _context_with_one_source()
    citations = build_citations(context, [1])
    assert len(citations) == 1
    assert citations[0].document_title == "Employee Security Policy"
    assert citations[0].chunk_id == "c1"


def test_build_citations_ignores_unknown_marker():
    context = _context_with_one_source()
    citations = build_citations(context, [1, 99])
    assert len(citations) == 1  # marker 99 silently dropped, never fabricated


def test_no_answer_response_is_trivially_grounded():
    context = BuiltContext(text="", sources=[])
    from app.models.schemas import CitationValidationResult
    citation_result = CitationValidationResult(valid=True, citations_checked=0, invalid_citations=[])
    grounding = evaluate_grounding(
        "What is our policy on unicorns?",
        "I couldn't find enough information in the internal documentation to answer this question.",
        context, citation_result,
    )
    assert grounding.supported is True
    assert grounding.has_sufficient_evidence is False


def test_grounding_fails_when_citations_invalid():
    context = _context_with_one_source()
    from app.models.schemas import CitationValidationResult
    bad_citation_result = CitationValidationResult(valid=False, citations_checked=1, invalid_citations=[9])
    grounding = evaluate_grounding(
        "What is the password policy?",
        "Passwords rotate every 90 days [9].",
        context, bad_citation_result,
    )
    assert grounding.supported is False
