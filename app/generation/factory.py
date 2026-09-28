from __future__ import annotations

from app.core.config import settings
from app.core.logging_config import get_logger, log_event
from app.generation.gemini_provider import GeminiProvider, GeminiUnavailableError
from app.generation.llm_provider import LLMProvider
from app.generation.mock_provider import MockProvider

logger = get_logger(__name__)


def get_llm_provider() -> LLMProvider:
    mode = settings.LLM_PROVIDER
    if mode == "mock":
        return MockProvider()
    if mode == "gemini":
        return GeminiProvider()
    # auto: use Gemini if a key is configured, otherwise fall back to the mock
    # provider so the system stays fully testable without network access.
    if settings.GEMINI_API_KEY:
        try:
            return GeminiProvider()
        except GeminiUnavailableError as e:
            log_event(logger, 30, "gemini_unavailable_falling_back_to_mock", error=str(e))
    return MockProvider()
