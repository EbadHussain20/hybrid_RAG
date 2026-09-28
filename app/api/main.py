"""FastAPI application exposing the RAG pipeline."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.indexer import build_indexes, load_retrievers, run_ingestion
from app.core.logging_config import configure_logging, get_logger, log_event, new_request_id
from app.core.rag_pipeline import RagPipeline
from app.ingestion.pipeline import FileTooLargeError, PathTraversalError
from app.ingestion.loaders import UnsupportedFileTypeError
from app.ingestion.tracking import get_document, list_documents
from app.models.schemas import QueryRequest, QueryResponse, SearchRequest

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="RAG Hybrid Search API",
    description="Production-grade RAG pipeline with hybrid (dense + BM25) retrieval over internal documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_state: dict = {"dense": None, "sparse": None, "pipeline": None, "start_time": time.time(),
                "query_count": 0, "last_index_stats": None}


def _ensure_pipeline_loaded() -> RagPipeline:
    if _state["pipeline"] is None:
        dense, sparse = load_retrievers()
        _state["dense"] = dense
        _state["sparse"] = sparse
        _state["pipeline"] = RagPipeline(dense, sparse)
    return _state["pipeline"]


@app.on_event("startup")
def _startup() -> None:
    try:
        _ensure_pipeline_loaded()
        log_event(logger, 20, "startup_index_loaded",
                  num_chunks=len(_state["dense"].chunks) if _state["dense"] else 0)
    except Exception as e:  # noqa: BLE001
        log_event(logger, 30, "startup_no_index_yet", error=str(e))


@app.get("/health")
def health():
    dense = _state.get("dense")
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _state["start_time"], 1),
        "index_loaded": bool(dense and dense.chunks),
        "num_indexed_chunks": len(dense.chunks) if dense else 0,
    }


@app.get("/metrics")
def metrics():
    dense = _state.get("dense")
    return {
        "uptime_seconds": round(time.time() - _state["start_time"], 1),
        "query_count": _state["query_count"],
        "num_indexed_chunks": len(dense.chunks) if dense else 0,
        "num_indexed_documents": len({c.document_id for c in dense.chunks}) if dense else 0,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "fusion_method": settings.FUSION_METHOD,
        "reranking_enabled": settings.RERANKING_ENABLED,
        "last_index_stats": _state["last_index_stats"],
    }


@app.post("/ingest")
def ingest_endpoint(force: bool = False):
    request_id = new_request_id()
    try:
        results = run_ingestion(force=force)
    except Exception as e:  # noqa: BLE001
        log_event(logger, 40, "ingest_endpoint_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e

    return {
        "request_id": request_id,
        "files_processed": len(results),
        "files_skipped_unchanged": sum(1 for r in results if r.skipped),
        "files_errored": sum(1 for r in results if r.error),
        "errors": [{"document_id": r.document_id, "error": r.error} for r in results if r.error],
    }


@app.post("/index")
def index_endpoint(force_full_rebuild: bool = False):
    request_id = new_request_id()
    try:
        stats = build_indexes(force_full_rebuild=force_full_rebuild)
    except Exception as e:  # noqa: BLE001
        log_event(logger, 40, "index_endpoint_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e

    _state["last_index_stats"] = stats
    # Reload retrievers/pipeline with the freshly built index.
    _state["pipeline"] = None
    _ensure_pipeline_loaded()

    return {"request_id": request_id, **stats}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    request_id = new_request_id()
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=422, detail="`question` must not be empty")

    pipeline = _ensure_pipeline_loaded()
    if not _state["dense"] or not _state["dense"].chunks:
        raise HTTPException(status_code=503,
                             detail="Index is empty. Call POST /ingest then POST /index first.")

    _state["query_count"] += 1
    response = pipeline.answer_question(
        req.question, top_k=req.top_k, reranking_enabled=req.reranking_enabled,
    )
    return response


@app.post("/search")
def search_endpoint(req: SearchRequest):
    request_id = new_request_id()
    pipeline = _ensure_pipeline_loaded()
    dense, sparse = _state["dense"], _state["sparse"]
    if not dense or not dense.chunks:
        raise HTTPException(status_code=503,
                             detail="Index is empty. Call POST /ingest then POST /index first.")

    if req.mode == "dense":
        results = dense.search(req.query, top_k=req.top_k)
        return {"request_id": request_id, "mode": "dense", "results": [
            {"chunk_id": c.chunk_id, "document_title": c.document_title, "section": c.section,
             "score": s, "rank": r, "content": c.content} for c, s, r in results]}
    if req.mode == "sparse":
        results = sparse.search(req.query, top_k=req.top_k)
        return {"request_id": request_id, "mode": "sparse", "results": [
            {"chunk_id": c.chunk_id, "document_title": c.document_title, "section": c.section,
             "score": s, "rank": r, "matched_terms": m, "content": c.content} for c, s, r, m in results]}

    candidates = pipeline.hybrid.retrieve(req.query, top_k=req.top_k)
    return {"request_id": request_id, "mode": "hybrid", "fusion_method": settings.FUSION_METHOD, "results": [
        {"chunk_id": c.chunk.chunk_id, "document_title": c.chunk.document_title, "section": c.chunk.section,
         "dense_score": c.dense_score, "sparse_score": c.sparse_score, "fusion_score": c.fusion_score,
         "rank": c.final_rank, "matched_terms": c.matched_terms, "content": c.chunk.content}
        for c in candidates]}


@app.get("/documents")
def documents_endpoint():
    return {"documents": list_documents()}


@app.get("/documents/{document_id}")
def document_detail_endpoint(document_id: str):
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    dense = _state.get("dense")
    chunks = [c for c in (dense.chunks if dense else []) if c.document_id == document_id]
    doc["num_chunks"] = len(chunks)
    doc["chunks"] = [{"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "section": c.section,
                       "content": c.content} for c in chunks]
    return doc


@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Upload a new document into the raw docs directory for later ingestion."""
    request_id = new_request_id()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

    safe_name = Path(file.filename or "upload").name  # strip any path components -> traversal protection
    dest = Path(settings.DOCS_DIR) / "uploads" / safe_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(contents)

    log_event(logger, 20, "file_uploaded", request_id=request_id, filename=safe_name, size_mb=round(size_mb, 2))
    return {"request_id": request_id, "filename": safe_name, "path": str(dest),
            "message": "Uploaded. Call POST /ingest then POST /index to include it in search."}


_frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")

    @app.get("/")
    def root():
        return FileResponse(str(_frontend_dir / "index.html"))
