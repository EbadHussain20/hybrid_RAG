"""Document parsing for supported file types.

Each loader returns raw extracted text plus a best-effort title and a list of
(section_heading, section_text) pairs so downstream chunking can be structure-aware.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    pass


class DocumentParseError(Exception):
    pass


@dataclass
class ParsedSection:
    heading: str | None
    text: str


@dataclass
class ParsedDocument:
    title: str
    document_type: str
    sections: list[ParsedSection] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())


def _split_markdown_like(text: str) -> list[ParsedSection]:
    """Split text on markdown-style headings (#, ##, ###) into sections."""
    lines = text.splitlines()
    sections: list[ParsedSection] = []
    current_heading = None
    buf: list[str] = []

    heading_re = re.compile(r"^(#{1,6})\s+(.*)")

    for line in lines:
        m = heading_re.match(line.strip())
        if m:
            if buf:
                sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
                buf = []
            current_heading = m.group(2).strip()
        else:
            buf.append(line)
    if buf:
        sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
    return [s for s in sections if s.text.strip()]


def load_markdown(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = _split_markdown_like(text)

    # The document title is the very first heading line in the file, independent
    # of whether that heading's own section has any body text below it (a lone
    # "# Title" line immediately followed by a "## Subheading" is common and
    # must not be dropped just because it has no body of its own).
    heading_re = re.compile(r"^(#{1,6})\s+(.*)")
    title = path.stem
    for line in text.splitlines():
        m = heading_re.match(line.strip())
        if m:
            title = m.group(2).strip()
            break

    return ParsedDocument(title=title, document_type="md", sections=sections)


def load_txt(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Plain text: split on blank-line-delimited paragraphs, no heading structure.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections = [ParsedSection(None, p) for p in paragraphs] or [ParsedSection(None, text)]
    title = path.stem.replace("_", " ").title()
    return ParsedDocument(title=title, document_type="txt", sections=sections)


def load_html(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    title = (title_tag.get_text(strip=True) if title_tag else None) or \
            (h1_tag.get_text(strip=True) if h1_tag else path.stem)

    sections: list[ParsedSection] = []
    current_heading = None
    buf: list[str] = []

    body = soup.body or soup
    for el in body.descendants:
        name = getattr(el, "name", None)
        if name in ("h1", "h2", "h3", "h4"):
            text = el.get_text(" ", strip=True)
            if text.lower() == (title or "").lower() and not sections and not buf:
                continue
            if buf:
                sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
                buf = []
            current_heading = text
        elif name in ("p", "li"):
            text = el.get_text(" ", strip=True)
            if text:
                buf.append(text)
    if buf:
        sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))

    if not sections:
        sections = [ParsedSection(None, soup.get_text(" ", strip=True))]

    return ParsedDocument(title=title, document_type="html", sections=[s for s in sections if s.text.strip()])


def load_docx(path: Path) -> ParsedDocument:
    doc = DocxDocument(str(path))
    sections: list[ParsedSection] = []
    current_heading = None
    buf: list[str] = []
    title = path.stem.replace("_", " ").title()
    first_heading_seen = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading") or style == "title":
            if not first_heading_seen and style in ("heading 1", "title"):
                title = text
                first_heading_seen = True
                continue
            if buf:
                sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
                buf = []
            current_heading = text
        else:
            buf.append(text)
    if buf:
        sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))

    if not sections:
        sections = [ParsedSection(None, "\n".join(p.text for p in doc.paragraphs))]

    return ParsedDocument(title=title, document_type="docx", sections=[s for s in sections if s.text.strip()])


def load_pdf(path: Path) -> ParsedDocument:
    try:
        reader = PdfReader(str(path))
    except Exception as e:  # noqa: BLE001
        raise DocumentParseError(f"Failed to open PDF {path}: {e}") from e

    full_text_pages = []
    for page in reader.pages:
        try:
            full_text_pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            full_text_pages.append("")
    full_text = "\n".join(full_text_pages)

    # PDFs rarely carry heading metadata via pypdf's text extraction; use heuristic:
    # a short line (<60 chars) followed by longer body text is treated as a heading.
    lines = [l.strip() for l in full_text.splitlines()]
    sections: list[ParsedSection] = []
    current_heading = None
    buf: list[str] = []
    title = path.stem.replace("_", " ").title()
    first_line_used_as_title = False

    for line in lines:
        if not line:
            continue
        is_heading_like = len(line) < 60 and not line.endswith((".", ",", ";")) and len(line.split()) <= 8
        if is_heading_like and not first_line_used_as_title:
            title = line
            first_line_used_as_title = True
            continue
        if is_heading_like:
            if buf:
                sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
                buf = []
            current_heading = line
        else:
            buf.append(line)
    if buf:
        sections.append(ParsedSection(current_heading, "\n".join(buf).strip()))
    if not sections:
        sections = [ParsedSection(None, full_text)]

    return ParsedDocument(title=title, document_type="pdf", sections=[s for s in sections if s.text.strip()])


_LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_txt,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
    ".pdf": load_pdf,
}


def parse_document(path: Path) -> ParsedDocument:
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")
    try:
        return loader(path)
    except (UnsupportedFileTypeError, DocumentParseError):
        raise
    except Exception as e:  # noqa: BLE001
        raise DocumentParseError(f"Failed to parse {path}: {e}") from e
