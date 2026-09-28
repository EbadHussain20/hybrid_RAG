from pathlib import Path

from app.ingestion.pipeline import ingest_file
from app.ingestion.cleaning import clean_text
from app.ingestion.loaders import parse_document, UnsupportedFileTypeError
import pytest


def test_clean_text_collapses_whitespace_and_hyphenation():
    raw = "This is reim-\nbursement   with   extra   spaces.\n\n\n\nNext para."
    cleaned = clean_text(raw)
    assert "reimbursement" in cleaned
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_parse_markdown_extracts_headings(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\n\n## Section One\n\nContent A.\n\n## Section Two\n\nContent B.\n")
    parsed = parse_document(md_path)
    assert parsed.title == "Title"
    headings = [s.heading for s in parsed.sections]
    assert "Section One" in headings
    assert "Section Two" in headings


def test_parse_html_extracts_title_and_paragraphs(tmp_path):
    html_path = tmp_path / "doc.html"
    html_path.write_text("<html><head><title>My Doc</title></head><body>"
                          "<h2>Heading A</h2><p>Body text here.</p></body></html>")
    parsed = parse_document(html_path)
    assert parsed.title == "My Doc"
    assert any("Body text here." in s.text for s in parsed.sections)


def test_parse_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "file.xyz"
    bad_path.write_text("content")
    with pytest.raises(UnsupportedFileTypeError):
        parse_document(bad_path)


def test_ingest_file_produces_chunks_with_metadata(tmp_path):
    docs_root = tmp_path / "docs" / "hr"
    docs_root.mkdir(parents=True)
    f = docs_root / "sample.md"
    f.write_text("# Sample Policy\n\n## Rules\n\nEmployees must do the thing every 30 days.\n")

    result = ingest_file(f, docs_root=tmp_path / "docs", force=True)
    assert result.error is None
    assert len(result.chunks) >= 1
    chunk = result.chunks[0]
    assert chunk.department == "hr"
    assert chunk.document_title == "Sample Policy"
    assert chunk.chunk_id
    assert chunk.document_id


def test_ingest_file_skips_unchanged_on_second_call(tmp_path):
    docs_root = tmp_path / "docs" / "hr"
    docs_root.mkdir(parents=True)
    f = docs_root / "sample.md"
    f.write_text("# Sample\n\nSome content that will not change.\n")

    first = ingest_file(f, docs_root=tmp_path / "docs", force=False)
    assert first.skipped is False
    second = ingest_file(f, docs_root=tmp_path / "docs", force=False)
    assert second.skipped is True
