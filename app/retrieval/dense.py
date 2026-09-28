"""Dense semantic retriever: cosine similarity over persisted embeddings."""
from __future__ import annotations

import numpy as np

from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.models.schemas import Chunk


class DenseRetriever:
    def __init__(self, provider: EmbeddingProvider | None = None):
        self.provider = provider or get_embedding_provider()
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.provider.fit(texts)
        self.embeddings = self.provider.encode(texts)

    def load(self, chunks: list[Chunk], embeddings: np.ndarray, provider: EmbeddingProvider) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.provider = provider

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float, int]]:
        """Returns list of (chunk, cosine_score, rank[1-indexed])."""
        if self.embeddings is None or not self.chunks:
            return []
        query_vec = self.provider.encode([query])[0]
        scores = self.embeddings @ query_vec  # embeddings are L2-normalized -> dot == cosine
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append((self.chunks[idx], float(scores[idx]), rank))
        return results
