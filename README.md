# RAG Hybrid Search over Internal Documents

A local-first RAG pipeline: ingestion → structure-aware chunking → dense + BM25 indexes → hybrid retrieval (RRF) → optional reranking → grounded generation → inline citations → citation validation → grounding check → evaluation.

## Architecture

```
 docs/ (pdf, docx, md, txt, html)
   │  loaders → cleaning → structure-aware chunking → metadata + deterministic IDs
   ▼
 ┌──────────────┐   ┌──────────────┐
 │ Dense index  │   │ BM25 index   │      (independent retrievers)
 │ TF-IDF+SVD   │   │ rank_bm25    │
 └──────┬───────┘   └──────┬───────┘
        └──────┬───────────┘
        Reciprocal Rank Fusion (or weighted)
               ▼
        optional reranker → context builder (dedupe, per-doc cap, token budget)
               ▼
        LLMProvider (Gemini | Mock) → answer with [n] markers
               ▼
        CitationValidator → grounding check → API / UI
```

## How hybrid retrieval works

Dense and sparse retrievers run independently and return ranked lists. Each candidate keeps `dense_score`, `sparse_score`, their ranks, and a `fusion_score`. Default fusion is **RRF**: `score = Σ 1/(RRF_K + rank)` across retrievers (`RRF_K=60`). RRF uses ranks, so incomparable score scales (cosine vs BM25) don't matter. `FUSION_METHOD=weighted` min-max normalizes both and combines with `DENSE_WEIGHT`/`SPARSE_WEIGHT`. A chunk found by only one retriever still gets scored.

## Chunking

Documents are split by native structure (headings), then paragraphs are packed up to `CHUNK_SIZE` characters with `CHUNK_OVERLAP` overlap. Oversized paragraphs are split on sentence boundaries. Each chunk keeps its section heading, used in citations.

## Citations and grounding

The prompt requires `[n]` markers tied to `[Source n]` blocks. `CitationValidator` checks every marker against the actual assembled context; invalid markers are flagged (`valid: false`, `invalid_citations`). Sources returned to the client are built only from real chunks. The grounding check is lexical (answer/context vocabulary overlap, citation validity, evidence present). It is a coarse proxy, **not** an LLM judge or a factual verifier.

## Setup and run

```bash
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY
python scripts/ingest.py
python scripts/index.py
python -m app                   # UI at http://localhost:8000, API docs at /docs
```

Other commands: `make test`, `make eval` (`python -m evaluation.run`), `docker compose up --build`.

Endpoints: `POST /ingest`, `POST /index`, `POST /query`, `POST /search` (mode: dense|sparse|hybrid), `POST /upload`, `GET /documents`, `GET /documents/{id}`, `GET /health`, `GET /metrics`.

## Measured results (30 questions, 26 with ground-truth sources)

| Metric | Dense | BM25 | Hybrid (RRF) |
|---|---|---|---|
| Recall@5 | 0.962 | 1.000 | 0.981 |
| Recall@3 | 0.923 | 0.923 | 0.942 |
| MRR | 0.904 | 0.904 | 0.886 |
| Hit rate | 0.962 | 1.000 | 1.000 |

Citation validity 100%, groundedness 100%, no-answer accuracy **50%** (2 of 4). Full table: `evaluation/results/report.md`.

**Honest reading:** on this small, keyword-friendly corpus, hybrid did *not* clearly beat BM25 (BM25 has the best Recall@5 and ties on MRR). Hybrid wins on Recall@3, Precision@5 and hit rate only marginally. With 26 questions, these differences are within noise. The dataset is too small to demonstrate hybrid's advantage.

## Verification status and deviations from the spec

Verified by execution: 42 passing tests, ingestion (all 5 formats), incremental indexing, dense/BM25/hybrid search, API endpoints against a live server, no-answer path, evaluation run.

Not verified / deviated:
- **Gemini was never called.** The sandbox could not reach Google's API. `GeminiProvider` is written against the REST contract but is untested against the live service. All results above used the deterministic `MockProvider` (an extractive stub), so answer quality, citation behavior, and no-answer accuracy with real Gemini are unknown. With no key set (`LLM_PROVIDER=auto`) the mock is used silently; check the `llm_provider` field in responses.
- **Dense embeddings are TF-IDF + SVD (LSA), not a neural model.** The sandbox lacked disk space and network for sentence-transformers. `SentenceTransformerEmbeddingProvider` exists (`EMBEDDING_PROVIDER=sentence_transformers`) but was never run.
- **Reranker is a lexical-overlap heuristic**, not a cross-encoder. `CrossEncoderReranker` exists but is untested.
- No LLM-as-judge evaluation; latency numbers are from the mock, so they exclude real LLM time.
- The mock false-answers 2 no-answer questions (volunteer time off, cryptocurrency) because adjacent terms overlap. A real LLM may behave differently.
- Docker files were written but not built or run.
- BM25 uses no stemming ("password" ≠ "passwords").

## Next steps

Run with a real Gemini key and a neural embedder, add a cross-encoder, expand the eval set (hundreds of questions) to measure hybrid properly, add an LLM judge, and swap in Qdrant/OpenSearch behind the retriever interfaces for scale.
