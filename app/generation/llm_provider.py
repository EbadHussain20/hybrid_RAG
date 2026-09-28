"""Abstract LLM provider so the generation backend can change without touching
the RAG pipeline (context building, citations, grounding all depend only on
this interface)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        ...
