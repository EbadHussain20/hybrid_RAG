"""Structured JSON logging with correlation IDs.

Never logs secrets (API keys) or raw document contents - only metadata,
latencies, and small identifiers, per the security requirements.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from app.core.config import settings

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_REDACT_KEYS = {"gemini_api_key", "api_key", "authorization", "content"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id_ctx.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            for k, v in extra.items():
                if k.lower() in _REDACT_KEYS:
                    continue
                payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id_ctx.set(rid)
    return rid


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields) -> None:
    logger.log(level, message, extra={"extra_fields": fields})


class Timer:
    """Small helper: `with Timer() as t: ...` then `t.ms`."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000.0
        return False
