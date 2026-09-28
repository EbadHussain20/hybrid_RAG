"""Abstract embedding provider interface, so the dense backend is swappable."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Fit any corpus-dependent transform (e.g. TF-IDF vocabulary). No-op for
        providers backed by pretrained models."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...
