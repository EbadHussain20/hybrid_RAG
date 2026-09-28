from app.chunking.chunker import chunk_sections


def test_chunking_respects_size_roughly():
    long_para = " ".join(["word"] * 300)  # ~1500 chars
    sections = [("Heading", long_para)]
    chunks = chunk_sections(sections, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 600  # allow small overlap slack


def test_chunking_preserves_heading():
    sections = [("Password Requirements", "Rotate passwords every 90 days.")]
    chunks = chunk_sections(sections, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].section == "Password Requirements"


def test_chunking_overlap_present_across_boundary():
    para1 = "A" * 400
    para2 = "B" * 400
    sections = [("H", f"{para1}\n\n{para2}")]
    chunks = chunk_sections(sections, chunk_size=450, chunk_overlap=100)
    assert len(chunks) >= 2
    # The tail of the first chunk should reappear at the start of the next (overlap).
    assert chunks[0].text[-50:] in chunks[1].text


def test_empty_section_produces_no_chunks():
    chunks = chunk_sections([(None, "   ")], chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_multiple_sections_produce_separate_chunk_groups():
    sections = [("A", "Short content A."), ("B", "Short content B.")]
    chunks = chunk_sections(sections, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 2
    assert chunks[0].section == "A"
    assert chunks[1].section == "B"
