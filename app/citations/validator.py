"""Extract citation markers from a generated answer and validate them against
the actual retrieved context - never trust the LLM's citation numbers blindly."""
from __future__ import annotations

import re

from app.generation.context_builder import BuiltContext
from app.models.schemas import Citation, CitationValidationResult

_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_citation_markers(answer_text: str) -> list[int]:
    """Extract unique citation marker numbers referenced in the answer body
    (ignores the trailing 'Sources:' list to avoid double counting)."""
    body = answer_text.split("\nSources:")[0]
    markers = [int(m) for m in _MARKER_RE.findall(body)]
    seen = []
    for m in markers:
        if m not in seen:
            seen.append(m)
    return seen


def build_citations(context: BuiltContext, used_markers: list[int]) -> list[Citation]:
    citations = []
    by_marker = {s.marker: s for s in context.sources}
    for marker in used_markers:
        source = by_marker.get(marker)
        if source is None:
            continue
        chunk = source.candidate.chunk
        snippet = chunk.content[:240] + ("..." if len(chunk.content) > 240 else "")
        citations.append(Citation(
            marker=marker,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            document_title=chunk.document_title,
            section=chunk.section,
            source_path=chunk.source_path,
            snippet=snippet,
        ))
    return citations


def validate_citations(answer_text: str, context: BuiltContext) -> CitationValidationResult:
    """Every citation marker in the answer must map to a real [Source N] block
    that was actually part of the retrieved/assembled context."""
    used_markers = extract_citation_markers(answer_text)
    valid_markers = {s.marker for s in context.sources}

    invalid = [m for m in used_markers if m not in valid_markers]
    details = []
    if invalid:
        details.append(f"Citation marker(s) {invalid} do not correspond to any retrieved source block.")
    if not used_markers and answer_text.strip() and "couldn't find enough information" not in answer_text:
        details.append("Answer contains no citation markers despite not being a no-answer response.")

    return CitationValidationResult(
        valid=len(invalid) == 0,
        citations_checked=len(used_markers),
        invalid_citations=invalid,
        details=details,
    )
