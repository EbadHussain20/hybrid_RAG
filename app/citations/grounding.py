"""Lightweight grounding validation, separate from the production answer path.

Checks, using evidence rather than a black-box judgment:
- citation validity (delegated to CitationValidationResult, already evidence-based)
- lexical overlap between the answer and the retrieved context (a crude but
  transparent proxy for "does this answer actually restate the sources or
  does it look like it drifted off into unsupported claims")
- relevance: overlap between the answer and the original question
- sufficiency: whether any context was retrieved at all
"""
from __future__ import annotations

from app.generation.context_builder import BuiltContext
from app.models.schemas import CitationValidationResult, GroundingResult
from app.retrieval.sparse import tokenize

NO_ANSWER_MARKER = "couldn't find enough information"


def evaluate_grounding(question: str, answer_text: str, context: BuiltContext,
                        citation_result: CitationValidationResult) -> GroundingResult:
    notes: list[str] = []

    if NO_ANSWER_MARKER in answer_text.lower():
        has_sufficient_evidence = len(context.sources) > 0
        notes.append("Model declined to answer (no-answer response); treated as trivially grounded.")
        return GroundingResult(
            supported=True,
            has_sufficient_evidence=has_sufficient_evidence,
            relevance_score=0.0,
            notes=notes,
        )

    answer_body = answer_text.split("\nSources:")[0]
    answer_terms = set(tokenize(answer_body))
    question_terms = set(tokenize(question))
    context_terms: set[str] = set()
    for s in context.sources:
        context_terms |= set(tokenize(s.candidate.chunk.content))

    if not answer_terms:
        notes.append("Answer body is empty.")
        return GroundingResult(supported=False, has_sufficient_evidence=False,
                                relevance_score=0.0, notes=notes)

    overlap_with_context = len(answer_terms & context_terms) / len(answer_terms)
    relevance_score = len(answer_terms & question_terms) / max(1, len(question_terms))

    supported = citation_result.valid and overlap_with_context >= 0.5
    has_sufficient_evidence = len(context.sources) > 0

    if not citation_result.valid:
        notes.append("Grounding failed: one or more citations do not map to retrieved sources.")
    if overlap_with_context < 0.5:
        notes.append(
            f"Only {overlap_with_context:.0%} of the answer's vocabulary overlaps with retrieved "
            "context; answer may contain unsupported claims."
        )
    if not has_sufficient_evidence:
        notes.append("No context was retrieved for this question.")

    return GroundingResult(
        supported=supported,
        has_sufficient_evidence=has_sufficient_evidence,
        relevance_score=round(relevance_score, 3),
        notes=notes,
    )
