"""Shared data models used across ingestion, retrieval, generation, and the API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    document_title: str
    source_path: str
    document_type: str  # pdf | docx | txt | md | html
    department: str
    section: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    content_hash: str
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source_path: str
    document_type: str
    department: str
    section: Optional[str] = None
    chunk_index: int
    content: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseModel):
    """A chunk plus its scores as it moves through the retrieval pipeline."""
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    fusion_score: float = 0.0
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None
    matched_terms: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    marker: int
    document_id: str
    chunk_id: str
    document_title: str
    section: Optional[str] = None
    source_path: str
    snippet: str


class CitationValidationResult(BaseModel):
    valid: bool
    citations_checked: int
    invalid_citations: list[int] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)


class GroundingResult(BaseModel):
    supported: bool
    has_sufficient_evidence: bool
    relevance_score: float
    notes: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    department_filter: Optional[str] = None
    reranking_enabled: Optional[bool] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "hybrid"  # dense | sparse | hybrid


class RetrievalDebugInfo(BaseModel):
    query: str
    dense_results: list[dict[str, Any]]
    sparse_results: list[dict[str, Any]]
    hybrid_results: list[dict[str, Any]]
    fusion_method: str
    reranking_applied: bool


class LatencyBreakdown(BaseModel):
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    sources: list[Citation]
    retrieval: RetrievalDebugInfo
    citation_validation: CitationValidationResult
    grounding: GroundingResult
    latency: LatencyBreakdown
    llm_provider: str
    model: str
