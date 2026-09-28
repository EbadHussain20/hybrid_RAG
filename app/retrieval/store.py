"""Persistent index storage.

Layout under settings.INDEX_DIR:
  chunks.json           - list of Chunk objects (metadata + content), the source of truth
  dense_embeddings.npy  - (N, dim) float32 matrix, row i corresponds to chunks[i]
  embedding_provider.pkl- fitted embedding provider state (vocabulary/SVD components)
  bm25_corpus.json      - tokenized corpus used to (re)build the BM25 index at load time
  documents.sqlite3      - document-level tracking (see app/ingestion/tracking.py)

Why these choices: everything is a plain file, so the whole index is
reproducible, diffable, and trivially backed up - no external database
service required. The migration path to a managed vector DB (Qdrant/Pinecone)
or Elasticsearch/OpenSearch for BM25 is: keep `chunks.json` as the canonical
metadata store and swap only the dense/sparse *retriever* implementations
(app/retrieval/dense.py, sparse.py) to call the managed service instead of
reading these local files - the rest of the pipeline (fusion, reranking,
generation, citations) is unaffected because it only depends on the
`RetrievalCandidate`/`Chunk` interfaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.models.schemas import Chunk

CHUNKS_PATH = Path(settings.INDEX_DIR) / "chunks.json"
EMBEDDINGS_PATH = Path(settings.INDEX_DIR) / "dense_embeddings.npy"
EMBEDDING_PROVIDER_PATH = Path(settings.INDEX_DIR) / "embedding_provider.pkl"
BM25_CORPUS_PATH = Path(settings.INDEX_DIR) / "bm25_corpus.json"


def save_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        json.dump([c.model_dump(mode="json") for c in chunks], f)


def load_chunks() -> list[Chunk]:
    if not CHUNKS_PATH.exists():
        return []
    with open(CHUNKS_PATH) as f:
        raw = json.load(f)
    return [Chunk(**r) for r in raw]


def save_dense_embeddings(embeddings: np.ndarray) -> None:
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)


def load_dense_embeddings() -> np.ndarray | None:
    if not EMBEDDINGS_PATH.exists():
        return None
    return np.load(EMBEDDINGS_PATH)


def index_exists() -> bool:
    return CHUNKS_PATH.exists() and EMBEDDINGS_PATH.exists()
