"""
Centralized configuration for the RAG pipeline.

All tunables are environment-variable driven so behavior can change without
code edits. Loaded once as a singleton `settings` object.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM ---
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    LLM_PROVIDER: str = "auto"  # auto | gemini | mock
    GEMINI_TIMEOUT_SECONDS: int = 30

    # --- Embeddings ---
    # "tfidf_svd": dependency-free local semantic embedding (default; works fully offline).
    # "sentence_transformers": swap in a real ST model when the runtime allows model downloads.
    EMBEDDING_PROVIDER: str = "tfidf_svd"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # used only if EMBEDDING_PROVIDER=sentence_transformers
    EMBEDDING_DIM: int = 256

    # --- Chunking ---
    CHUNK_SIZE: int = 800          # characters
    CHUNK_OVERLAP: int = 150       # characters

    # --- Retrieval ---
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    HYBRID_TOP_K: int = 8
    FUSION_METHOD: str = "rrf"     # rrf | weighted
    RRF_K: int = 60
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4

    # --- Reranking ---
    RERANKING_ENABLED: bool = True
    RERANK_TOP_K: int = 5
    RERANKER_BACKEND: str = "lexical_overlap"  # lexical_overlap | cross_encoder

    # --- Context management ---
    MAX_CONTEXT_TOKENS: int = 1800
    MAX_CHUNKS_PER_DOCUMENT: int = 2

    # --- Paths ---
    DATA_DIR: str = str(PROJECT_ROOT / "data")
    DOCS_DIR: str = str(PROJECT_ROOT / "docs")
    INDEX_DIR: str = str(PROJECT_ROOT / "data" / "index")

    # --- Security / limits ---
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.md,.html,.htm"

    # --- Server ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Ensure runtime directories exist.
for _dir in (settings.DATA_DIR, settings.INDEX_DIR,
             os.path.join(settings.DATA_DIR, "raw"),
             os.path.join(settings.DATA_DIR, "processed")):
    Path(_dir).mkdir(parents=True, exist_ok=True)
