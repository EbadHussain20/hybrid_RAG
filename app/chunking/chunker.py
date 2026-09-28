"""Structure-aware chunking.

Strategy (documented in README):
1. Documents are first parsed into sections using their native structure
   (markdown/HTML headings, DOCX heading styles, or heuristic heading detection
   for PDF/TXT) - see app/ingestion/loaders.py.
2. Each section is then split into paragraphs.
3. Paragraphs are packed into chunks up to CHUNK_SIZE characters, never splitting
   a paragraph unless the paragraph alone exceeds CHUNK_SIZE (in which case it is
   hard-split on sentence boundaries).
4. Consecutive chunks within the same section overlap by CHUNK_OVERLAP characters
   so context isn't lost across a chunk boundary.
5. Each chunk retains the section heading it came from, which flows into citation
   metadata (e.g. "Employee Security Policy - Password Requirements").

This avoids splitting a policy's password rule from its surrounding paragraph
just because it landed mid-sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import settings

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RawChunk:
    section: str | None
    text: str


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def _hard_split_long_paragraph(paragraph: str, max_len: int) -> list[str]:
    """Split an overlong paragraph on sentence boundaries, packing greedily."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) > max_len and buf:
            chunks.append(buf)
            buf = sent
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    # Extreme edge case: a single sentence longer than max_len - hard character split.
    final: list[str] = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            for i in range(0, len(c), max_len):
                final.append(c[i:i + max_len])
    return final


def chunk_sections(sections: list[tuple[str | None, str]],
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[RawChunk]:
    """Pack (heading, cleaned_text) sections into overlapping chunks."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    raw_chunks: list[RawChunk] = []

    for heading, section_text in sections:
        if not section_text.strip():
            continue
        paragraphs = _split_paragraphs(section_text)

        # Pre-split any paragraph that alone exceeds chunk_size.
        units: list[str] = []
        for p in paragraphs:
            if len(p) > chunk_size:
                units.extend(_hard_split_long_paragraph(p, chunk_size))
            else:
                units.append(p)

        buf = ""
        for unit in units:
            candidate = f"{buf}\n\n{unit}".strip() if buf else unit
            if len(candidate) > chunk_size and buf:
                raw_chunks.append(RawChunk(heading, buf))
                # Start new buffer with overlap tail of previous buffer.
                overlap_tail = buf[-chunk_overlap:] if chunk_overlap > 0 else ""
                buf = f"{overlap_tail}\n\n{unit}".strip() if overlap_tail else unit
            else:
                buf = candidate
        if buf:
            raw_chunks.append(RawChunk(heading, buf))

    return raw_chunks
