"""Gemini provider, calling the official Generative Language REST API directly
(no SDK dependency needed beyond `requests`). Reads the API key only from the
environment (GEMINI_API_KEY) - never hardcoded, never logged.

Note: this sandbox's outbound network allowlist does not include
generativelanguage.googleapis.com, so this class could not be exercised
end-to-end during this build (see README "Verification" section for exactly
what was and wasn't run live). The MockProvider is used for all tests. This
class is nonetheless a complete, real implementation of the Gemini call
contract and will work as soon as GEMINI_API_KEY is set in an environment
that can reach Google's API.
"""
from __future__ import annotations

import requests

from app.core.config import settings
from app.core.logging_config import get_logger, log_event
from app.generation.llm_provider import LLMProvider, LLMResponse

logger = get_logger(__name__)

_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiUnavailableError(Exception):
    pass


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        if not self.api_key:
            raise GeminiUnavailableError("GEMINI_API_KEY is not set")

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = _ENDPOINT_TEMPLATE.format(model=self.model)
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        try:
            resp = requests.post(url, json=payload, headers=headers,
                                  timeout=settings.GEMINI_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            log_event(logger, 40, "gemini_request_failed", error=str(e))
            return LLMResponse(text="", model=self.model, provider=self.name, error=str(e))

        if resp.status_code == 429:
            log_event(logger, 30, "gemini_rate_limited")
            return LLMResponse(text="", model=self.model, provider=self.name,
                                error="Gemini API rate limit exceeded")
        if resp.status_code >= 400:
            log_event(logger, 40, "gemini_error_status", status=resp.status_code)
            return LLMResponse(text="", model=self.model, provider=self.name,
                                error=f"Gemini API error: HTTP {resp.status_code}")

        data = resp.json()
        try:
            candidates = data["candidates"]
            text = "".join(p.get("text", "") for p in candidates[0]["content"]["parts"])
            usage = data.get("usageMetadata", {})
            return LLMResponse(
                text=text, model=self.model, provider=self.name,
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
            )
        except (KeyError, IndexError) as e:
            log_event(logger, 40, "gemini_unexpected_response_shape", error=str(e))
            return LLMResponse(text="", model=self.model, provider=self.name,
                                error="Unexpected Gemini response shape")
