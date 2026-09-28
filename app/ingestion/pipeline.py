"""End-to-end ingestion: filesystem -> ParsedDocument -> cleaned sections -> Chunks.

Deterministic IDs: document_id = sha1(source_path); chunk_id = sha1(document_id:chunk_index).
This means re-ingesting unchanged files produces identical IDs, which is what
makes incremental indexing (skip unchanged documents) and stable citations possible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.chunking.chunker import chunk_sections
from app.core.config import settings
from app.core.logging_config import get_logger, log_event
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import DocumentParseError, UnsupportedFileTypeError, parse_document
from app.ingestion.tracking import get_known_hash, upsert_document
from app.models.schemas import Chunk

logger = get_logger(__name__)


class FileTooLargeError(Exception):
    pass


class PathTraversalError(Exception):
    pass


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _document_id_for(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _chunk_id_for(document_id: str, index: int) -> str:
    return hashlib.sha1(f"{document_id}:{index}".encode("utf-8")).hexdigest()[:16]


def infer_department(path: Path, docs_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(docs_root.resolve())
        return rel.parts[0] if rel.parts else "general"
    except ValueError:
        return "general"


def _validate_file(path: Path) -> None:
    resolved = path.resolve()
    docs_root = Path(settings.DOCS_DIR).resolve()
    if docs_root not in resolved.parents and resolved != docs_root:
        # Allow files outside DOCS_DIR too (e.g. ad-hoc uploads), but block traversal tricks.
        if ".." in path.parts:
            raise PathTraversalError(f"Rejected path with traversal segment: {path}")
    size_mb = resolved.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise FileTooLargeError(f"{path} is {size_mb:.1f}MB, exceeds MAX_FILE_SIZE_MB={settings.MAX_FILE_SIZE_MB}")
    if resolved.suffix.lower() not in settings.allowed_extensions_set:
        raise UnsupportedFileTypeError(f"Extension {resolved.suffix} not permitted")


@dataclass
class IngestResult:
    document_id: str
    title: str
    department: str
    chunks: list[Chunk]
    skipped: bool  # True if unchanged since last ingest
    error: str | None = None


def ingest_file(path: Path, docs_root: Path | None = None, force: bool = False) -> IngestResult:
    docs_root = docs_root or Path(settings.DOCS_DIR)
    document_id = _document_id_for(path)

    try:
        _validate_file(path)
        parsed = parse_document(path)
    except (UnsupportedFileTypeError, DocumentParseError, FileTooLargeError, PathTraversalError) as e:
        log_event(logger, 40, "ingest_error", path=str(path), error=str(e))
        return IngestResult(document_id=document_id, title=path.stem, department="unknown",
                             chunks=[], skipped=False, error=str(e))

    cleaned_sections = [(s.heading, clean_text(s.text)) for s in parsed.sections]
    full_text_for_hash = "\n".join(t for _, t in cleaned_sections)
    content_hash = _hash_text(full_text_for_hash)

    if not force:
        known_hash = get_known_hash(document_id)
        if known_hash == content_hash:
            log_event(logger, 20, "ingest_skip_unchanged", path=str(path), document_id=document_id)
            return IngestResult(document_id=document_id, title=parsed.title,
                                 department=infer_department(path, docs_root), chunks=[], skipped=True)

    department = infer_department(path, docs_root)
    now = datetime.now(timezone.utc).isoformat()

    raw_chunks = chunk_sections([(h, t) for h, t in cleaned_sections])

    chunks: list[Chunk] = []
    for idx, rc in enumerate(raw_chunks):
        chunk_id = _chunk_id_for(document_id, idx)
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=parsed.title,
            source_path=str(path),
            document_type=parsed.document_type,
            department=department,
            section=rc.section,
            chunk_index=idx,
            content=rc.text,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            metadata={"char_count": len(rc.text)},
        ))

    upsert_document(document_id, parsed.title, str(path), parsed.document_type, department,
                     content_hash, now, now)

    log_event(logger, 20, "ingest_success", path=str(path), document_id=document_id,
              num_chunks=len(chunks), department=department)

    return IngestResult(document_id=document_id, title=parsed.title, department=department,
                         chunks=chunks, skipped=False)


def discover_files(root: Path) -> list[Path]:
    exts = settings.allowed_extensions_set
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def ingest_directory(root: Path | None = None, force: bool = False) -> list[IngestResult]:
    root = root or Path(settings.DOCS_DIR)
    files = discover_files(root)
    results = []
    for f in files:
        results.append(ingest_file(f, docs_root=root, force=force))
    return results
