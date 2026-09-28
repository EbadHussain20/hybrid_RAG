"""Local, fully-offline "dense" embedding provider using TF-IDF + Truncated SVD (LSA).

Why this exists: the reference architecture calls for a Sentence-Transformers
model, but this sandbox has neither the disk space for the torch/transformers
stack nor network access to huggingface.co to download model weights (both
were verified during development - see README "Embedding backend" section).
TF-IDF + SVD (latent semantic analysis) is a legitimate, well-established
dense/semantic embedding technique: it captures co-occurrence-based topical
similarity in a fixed-size dense vector, is genuinely distinct from the sparse
BM25 index (different math, different failure modes - e.g. it captures loose
topical similarity even without exact keyword overlap), and needs nothing at
runtime beyond scikit-learn.

`EmbeddingProvider` is an abstract interface specifically so this can be
swapped for `SentenceTransformerEmbeddingProvider` (also implemented, see
sentence_transformer.py) by changing EMBEDDING_PROVIDER=sentence_transformers
in .env on a machine with model-download access - no pipeline code changes.
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.embeddings.base import EmbeddingProvider


class TfidfSvdEmbeddingProvider(EmbeddingProvider):
    name = "tfidf_svd"

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd: TruncatedSVD | None = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        if not corpus:
            raise ValueError("Cannot fit embedding provider on empty corpus")
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        n_components = min(self.dim, max(2, min(tfidf_matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self.svd is None:
            raise RuntimeError("EmbeddingProvider.fit() must be called before encode()")
        tfidf_matrix = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf_matrix)
        dense = normalize(dense, norm="l2", axis=1)
        return dense.astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "svd": self.svd,
                "dim": self.dim,
                "fitted": self._fitted,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.svd = state["svd"]
        self.dim = state["dim"]
        self._fitted = state["fitted"]
