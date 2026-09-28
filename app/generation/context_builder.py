"""Assemble retrieved candidates into a bounded, formatted context for the LLM."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.models.schemas import RetrievalCandidate

# Rough token estimate without pulling in a tokenizer dependency: ~4 chars/token in English.
_CHARS_PER_TOKEN = 4


@dataclass
class ContextSource:
    marker: int
    candidate: RetrievalCandidate


@dataclass
class BuiltContext:
    text: str
    sources: list[ContextSource]


def build_context(candidates: list[RetrievalCandidate],
                   max_context_tokens: int | None = None,
                   max_per_document: int | None = None) -> BuiltContext:
    max_context_tokens = max_context_tokens or settings.MAX_CONTEXT_TOKENS
    max_per_document = max_per_document or settings.MAX_CHUNKS_PER_DOCUMENT
    max_chars = max_context_tokens * _CHARS_PER_TOKEN

    seen_chunk_ids: set[str] = set()
    per_doc_count: dict[str, int] = {}
    selected: list[RetrievalCandidate] = []

    for cand in candidates:
        cid = cand.chunk.chunk_id
        doc_id = cand.chunk.document_id
        if cid in seen_chunk_ids:
            continue
        if per_doc_count.get(doc_id, 0) >= max_per_document:
            continue
        seen_chunk_ids.add(cid)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
        selected.append(cand)

    blocks = []
    sources: list[ContextSource] = []
    running_chars = 0

    for i, cand in enumerate(selected, start=1):
        chunk = cand.chunk
        header = f"[Source {i}]\nDocument: {chunk.document_title}"
        if chunk.section:
            header += f"\nSection: {chunk.section}"
        header += f"\nChunk ID: {chunk.chunk_id}"
        block = f"{header}\n\n{chunk.content}\n"

        if running_chars + len(block) > max_chars and blocks:
            break

        blocks.append(block)
        sources.append(ContextSource(marker=i, candidate=cand))
        running_chars += len(block)

    return BuiltContext(text="\n---\n".join(blocks), sources=sources)
