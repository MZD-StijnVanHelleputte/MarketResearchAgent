import pytest
from retrieval.chunker import Chunker


def test_chunk_text_produces_correct_overlap():
    chunker = Chunker(chunk_size=5, chunk_overlap=2)
    words = list(range(12))
    text = " ".join(str(w) for w in words)
    chunks = chunker.chunk_text(text, source="test", domain="mining")

    assert len(chunks) > 1
    # Each chunk except the last should have chunk_size words
    first = chunks[0].text.split()
    assert len(first) == 5
    # Second chunk should start with the overlap from the end of the first
    second = chunks[1].text.split()
    assert second[:2] == first[3:]  # overlap = 2, so last 2 words of first == first 2 of second


def test_chunk_text_single_chunk_when_text_short():
    chunker = Chunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_text("hello world", source="s", domain="d")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunk_text_empty_returns_empty():
    chunker = Chunker()
    assert chunker.chunk_text("", source="s", domain="d") == []
    assert chunker.chunk_text("   ", source="s", domain="d") == []


def test_chunk_as_one_single_document():
    chunker = Chunker()
    chunks = chunker.chunk_as_one("some text here", source="src", domain="dom")
    assert len(chunks) == 1
    assert chunks[0].text == "some text here"
    assert chunks[0].source == "src"
    assert chunks[0].domain == "dom"


def test_chunk_as_one_truncates_at_10k_chars():
    chunker = Chunker()
    long_text = "word " * 3000  # ~15000 chars
    chunks = chunker.chunk_as_one(long_text, source="s", domain="d")
    assert len(chunks) == 1
    assert len(chunks[0].text) <= 10_000


def test_chunk_document_splits_on_headers():
    chunker = Chunker(chunk_size=50, chunk_overlap=5)
    text = "# Section A\n\nContent of section A with enough words.\n\n## Section B\n\nContent of section B.\n"
    chunks = chunker.chunk_document(text, source="doc.md", domain="mining")
    assert len(chunks) >= 2
    # Each chunk should contain content from a section
    all_text = " ".join(c.text for c in chunks)
    assert "Section A" in all_text or "Content of section A" in all_text
    assert "Content of section B" in all_text


def test_chunk_document_falls_back_to_chunk_text_when_no_headers():
    chunker = Chunker(chunk_size=5, chunk_overlap=1)
    text = "one two three four five six seven eight"
    chunks = chunker.chunk_document(text, source="plain.md", domain="construction")
    assert len(chunks) >= 1


def test_chunk_text_caps_pathological_whitespace_free_run():
    """A single huge run with no whitespace (e.g. glued PDF table extraction)
    must still be hard-split by byte length, even though it counts as just
    one 'word' (the failure mode a word-count-only cap missed)."""
    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    text = "1234567890,./;'" * 20_000
    chunks = chunker.chunk_text(text, source="s", domain="d")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.encode("utf-8")) <= Chunker._MAX_BYTES


def test_chunk_document_caps_pathological_run_inside_section():
    chunker = Chunker(chunk_size=600, chunk_overlap=100)
    text = "# Section\n\n" + ("1234567890,./;'" * 10_000)
    chunks = chunker.chunk_document(text, source="doc.md", domain="mining")
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.encode("utf-8")) <= Chunker._MAX_BYTES


def test_each_chunk_has_unique_chunk_id_and_timestamp():
    chunker = Chunker(chunk_size=3, chunk_overlap=1)
    chunks = chunker.chunk_text("a b c d e f g", source="s", domain="d")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"
    for c in chunks:
        assert c.timestamp  # non-empty ISO timestamp
