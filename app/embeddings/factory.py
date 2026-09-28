from __future__ import annotations

from app.core.config import settings
from app.embeddings.base import EmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
        return SentenceTransformerEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
    from app.embeddings.tfidf_svd import TfidfSvdEmbeddingProvider
    return TfidfSvdEmbeddingProvider(dim=settings.EMBEDDING_DIM)
