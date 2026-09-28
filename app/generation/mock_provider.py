"""Deterministic mock LLM provider.

Used automatically when GEMINI_API_KEY is unset/unreachable, and always used
in the test suite. It does not call any external service. It performs a
genuine (if simple) extractive strategy: it pulls the sentences most relevant
to the question from each provided [Source N] block and cites them with that
block's real marker - so it exercises the exact same citation contract a real
LLM answer would, including correctly producing the "no answer" response when
no source block is relevant enough.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.generation.llm_provider import LLMProvider, LLMResponse
from app.retrieval.sparse import tokenize

_SOURCE_BLOCK_RE = re.compile(
    r"\[Source (\d+)\]\nDocument: (.*?)(?:\nSection: (.*?))?\nChunk ID: .*?\n\n(.*?)(?=\n---\n|\Z)",
    re.DOTALL,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

NO_ANSWER_TEXT = (
    "I couldn't find enough information in the internal documentation to answer this question."
)

# Ultra-generic domain terms that appear across almost every internal document and
# so are poor signals of true topical relevance for this simple extractive mock
# (a real LLM does this discrimination implicitly; BM25 handles it via IDF).
_GENERIC_TERMS = ENGLISH_STOP_WORDS | {
    "policy", "policies", "company", "companys", "employee", "employees",
    "department", "process", "requires", "required", "approval",
}


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock-extractive-v1"):
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        q_match = re.search(r"Question:\s*(.*)", user_prompt)
        question = q_match.group(1).strip() if q_match else ""
        question_terms = set(tokenize(question)) - _GENERIC_TERMS
        if not question_terms:
            question_terms = set(tokenize(question))  # fall back if the question was all-generic

        context_match = re.search(r"Retrieved internal documentation sources:\n\n(.*?)\n\nWrite a grounded answer",
                                   user_prompt, re.DOTALL)
        context_text = context_match.group(1) if context_match else ""

        blocks = _SOURCE_BLOCK_RE.findall(context_text)
        scored_sentences: list[tuple[float, str, int, str, str]] = []  # score, sentence, marker, title, section

        # Require at least 2 overlapping content terms (or all terms for very short
        # questions) so passing mentions of a policy's name don't outscore sentences
        # that actually state the policy content. Earlier source blocks are the
        # retriever's top hybrid/rerank picks, so they get a small preference too.
        min_overlap = 2 if len(question_terms) >= 3 else 1

        for marker_str, title, section, content in blocks:
            marker = int(marker_str)
            rank_bonus = 1.0 / marker
            sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(content) if s.strip()]
            for sent in sentences:
                sent_terms = set(tokenize(sent))
                overlap = len(question_terms & sent_terms)
                if overlap < min_overlap:
                    continue
                coverage = overlap / max(1, len(question_terms))
                score = coverage + 0.15 * rank_bonus
                scored_sentences.append((score, sent, marker, title.strip(), (section or "").strip()))

        scored_sentences.sort(key=lambda t: -t[0])
        top_sentences = scored_sentences[:4]

        if not top_sentences:
            return LLMResponse(text=NO_ANSWER_TEXT, model=self.model, provider=self.name)

        used_markers: dict[int, tuple[str, str]] = {}
        answer_lines = []
        for _, sent, marker, title, section in top_sentences:
            if not sent.endswith((".", "!", "?")):
                sent += "."
            answer_lines.append(f"{sent} [{marker}]")
            used_markers[marker] = (title, section)

        sources_lines = ["", "Sources:"]
        for marker in sorted(used_markers):
            title, section = used_markers[marker]
            label = f"[{marker}] {title}"
            if section:
                label += f" — {section}"
            sources_lines.append(label)

        answer = " ".join(answer_lines) + "\n" + "\n".join(sources_lines)
        return LLMResponse(text=answer, model=self.model, provider=self.name)
