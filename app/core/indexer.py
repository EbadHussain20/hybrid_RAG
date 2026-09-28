"""Builds and persists the dense + sparse indexes from ingested chunks.

Incremental indexing: `ingest_directory` already skips re-parsing/re-chunking
unchanged files (via content hash in tracking.py). This module additionally
merges freshly-changed chunks with previously-indexed chunks for documents
that didn't change, so a full re-embed of the whole corpus is only needed the
first time or when the embedding provider/config changes.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.logging_config import get_logger, log_event
from app.embeddings.factory import get_embedding_provider
from app.ingestion.pipeline import IngestResult, ingest_directory
from app.models.schemas import Chunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.store import (
    EMBEDDING_PROVIDER_PATH,
    load_chunks,
    load_dense_embeddings,
    save_chunks,
    save_dense_embeddings,
)

logger = get_logger(__name__)


def run_ingestion(force: bool = False) -> list[IngestResult]:
    docs_root = Path(settings.DOCS_DIR)
    results = ingest_directory(docs_root, force=force)
    log_event(logger, 20, "ingestion_complete", num_files=len(results),
              num_skipped=sum(1 for r in results if r.skipped),
              num_errors=sum(1 for r in results if r.error))
    return results


def build_indexes(ingest_results: list[IngestResult] | None = None,
                   force_full_rebuild: bool = False) -> dict:
    """Merge new/changed chunks with existing index and rebuild dense+sparse indexes."""
    existing_chunks = [] if force_full_rebuild else load_chunks()

    # If there is no persisted index yet, the content-hash check in ingest_file
    # would otherwise report every unchanged-since-last-run file as "skipped"
    # without returning its chunks (chunks are only regenerated on change).
    # Force a full parse/chunk pass in that case so we actually have content to index.
    need_full_parse = force_full_rebuild or not existing_chunks
    ingest_results = ingest_results if ingest_results is not None else run_ingestion(force=need_full_parse)
    existing_by_doc: dict[str, list[Chunk]] = {}
    for c in existing_chunks:
        existing_by_doc.setdefault(c.document_id, []).append(c)

    changed_doc_ids = {r.document_id for r in ingest_results if not r.skipped and not r.error}

    final_chunks: list[Chunk] = []
    for r in ingest_results:
        if r.error:
            continue
        if r.skipped:
            final_chunks.extend(existing_by_doc.get(r.document_id, []))
        else:
            final_chunks.extend(r.chunks)

    if not final_chunks:
        log_event(logger, 30, "build_indexes_no_chunks")
        return {"num_chunks": 0, "num_documents": 0, "changed_documents": 0}

    # With a corpus-fit embedding provider (TF-IDF+SVD), any vocabulary change
    # requires refitting on the full corpus for consistent vector space - so we
    # always re-encode all chunks, but skip re-parsing/re-chunking unchanged
    # documents (the expensive I/O-bound step) via the hash check above.
    dense = DenseRetriever()
    dense.build(final_chunks)
    dense.provider.save(str(EMBEDDING_PROVIDER_PATH))

    sparse = SparseRetriever()
    sparse.build(final_chunks)

    save_chunks(final_chunks)
    save_dense_embeddings(dense.embeddings)

    log_event(logger, 20, "build_indexes_complete", num_chunks=len(final_chunks),
              num_documents=len({c.document_id for c in final_chunks}),
              changed_documents=len(changed_doc_ids))

    return {
        "num_chunks": len(final_chunks),
        "num_documents": len({c.document_id for c in final_chunks}),
        "changed_documents": len(changed_doc_ids),
    }


def load_retrievers() -> tuple[DenseRetriever, SparseRetriever]:
    chunks = load_chunks()
    embeddings = load_dense_embeddings()

    provider = get_embedding_provider()
    if EMBEDDING_PROVIDER_PATH.exists():
        provider.load(str(EMBEDDING_PROVIDER_PATH))

    dense = DenseRetriever(provider=provider)
    if chunks and embeddings is not None:
        dense.load(chunks, embeddings, provider)

    sparse = SparseRetriever()
    if chunks:
        sparse.load(chunks)

    return dense, sparse
