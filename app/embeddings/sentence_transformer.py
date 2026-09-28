"""Optional Sentence-Transformers backed embedding provider.

Not used by default in this sandbox (see tfidf_svd.py docstring for why), but
kept fully implemented: set EMBEDDING_PROVIDER=sentence_transformers and
EMBEDDING_MODEL=<model name> in .env on a machine with internet access to
huggingface.co, and app/embeddings/factory.py will use this instead - no other
pipeline code changes needed.
"""
from __future__ import annotations

import json

import numpy as np

from app.embeddings.base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = 0

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # local import: heavy dep
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def fit(self, corpus: list[str]) -> None:
        # Pretrained model - nothing corpus-dependent to fit, just lazy-load.
        self._load_model()

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            texts, batch_size=32, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def save(self, path: str) -> None:
        # Only the model name needs to persist - weights are re-downloaded/cached by name.
        with open(path, "w") as f:
            json.dump({"model_name": self.model_name}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        self.model_name = state["model_name"]
        self._model = None
