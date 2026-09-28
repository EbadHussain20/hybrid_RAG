"""Text cleaning applied to parsed document sections before chunking."""
from __future__ import annotations

import re

_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(text: str) -> str:
    """Normalize whitespace, strip control characters, and de-hyphenate line-wraps."""
    if not text:
        return ""
    text = _CONTROL_CHARS_RE.sub("", text)
    # Fix hyphenated line breaks like "reim-\nbursement" -> "reimbursement"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = _MULTI_WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return text
