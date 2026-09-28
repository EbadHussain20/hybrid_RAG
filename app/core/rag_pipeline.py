"""Orchestrates the full retrieval -> generation -> citation -> grounding pipeline."""
from __future__ import annotations

import uuid

from app.citations.grounding import evaluate_grounding
from app.citations.validator import build_citations, extract_citation_markers, validate_citations
from app.core.config import settings
from app.core.logging_config import Timer, get_logger, log_event
from app.generation.context_builder import build_context
from app.generation.factory import get_llm_provider
from app.generation.prompt_builder import SYSTEM_PROMPT, build_user_prompt
from app.models.schemas import (
    LatencyBreakdown,
    QueryResponse,
    RetrievalDebugInfo,
)
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.retrieval.sparse import SparseRetriever

logger = get_logger(__name__)


class RagPipeline:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse
        self.hybrid = HybridRetriever(dense, sparse)
        self.reranker = get_reranker()

    def answer_question(self, question: str, top_k: int | None = None,
                         reranking_enabled: bool | None = None) -> QueryResponse:
        request_id = uuid.uuid4().hex[:12]
        top_k = top_k or settings.RERANK_TOP_K
        rerank_on = settings.RERANKING_ENABLED if reranking_enabled is None else reranking_enabled

        latency = LatencyBreakdown()

        with Timer() as t_dense:
            dense_raw = self.dense.search(question, top_k=settings.DENSE_TOP_K)
        latency.dense_retrieval_ms = t_dense.ms

        with Timer() as t_sparse:
            sparse_raw = self.sparse.search(question, top_k=settings.SPARSE_TOP_K)
        latency.sparse_retrieval_ms = t_sparse.ms

        with Timer() as t_fusion:
            candidates = self.hybrid.retrieve(question, top_k=settings.HYBRID_TOP_K)
        latency.fusion_ms = t_fusion.ms

        reranking_applied = False
        if rerank_on and candidates:
            with Timer() as t_rerank:
                candidates = self.reranker.rerank(question, candidates, top_k=top_k)
            latency.reranking_ms = t_rerank.ms
            reranking_applied = True
        else:
            candidates = candidates[:top_k]
            for i, c in enumerate(candidates, start=1):
                c.final_rank = i

        context = build_context(candidates)

        with Timer() as t_llm:
            llm = get_llm_provider()
            user_prompt = build_user_prompt(question, context.text)
            llm_response = llm.generate(SYSTEM_PROMPT, user_prompt)
        latency.llm_ms = t_llm.ms

        answer_text = llm_response.text or (
            "I couldn't find enough information in the internal documentation to answer this question."
        )

        citation_result = validate_citations(answer_text, context)
        used_markers = extract_citation_markers(answer_text)
        sources = build_citations(context, used_markers)
        grounding = evaluate_grounding(question, answer_text, context, citation_result)

        latency.total_ms = (latency.dense_retrieval_ms + latency.sparse_retrieval_ms +
                             latency.fusion_ms + latency.reranking_ms + latency.llm_ms)

        retrieval_debug = RetrievalDebugInfo(
            query=question,
            dense_results=[{
                "chunk_id": c.chunk_id, "document_title": c.document_title, "score": s, "rank": r,
            } for c, s, r in dense_raw],
            sparse_results=[{
                "chunk_id": c.chunk_id, "document_title": c.document_title, "score": s, "rank": r,
                "matched_terms": m,
            } for c, s, r, m in sparse_raw],
            hybrid_results=[{
                "chunk_id": c.chunk.chunk_id, "document_title": c.chunk.document_title,
                "dense_score": c.dense_score, "sparse_score": c.sparse_score,
                "fusion_score": c.fusion_score, "rerank_score": c.rerank_score, "rank": c.final_rank,
            } for c in candidates],
            fusion_method=settings.FUSION_METHOD,
            reranking_applied=reranking_applied,
        )

        log_event(logger, 20, "query_answered", request_id=request_id,
                  num_candidates=len(candidates), citations_valid=citation_result.valid,
                  grounded=grounding.supported, total_ms=round(latency.total_ms, 1),
                  llm_provider=llm_response.provider)

        return QueryResponse(
            request_id=request_id,
            question=question,
            answer=answer_text,
            sources=sources,
            retrieval=retrieval_debug,
            citation_validation=citation_result,
            grounding=grounding,
            latency=latency,
            llm_provider=llm_response.provider,
            model=llm_response.model,
        )
