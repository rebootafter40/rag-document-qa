"""Tests for the text chunking module."""

import pytest
from src.chunker import chunk_text


def make_pages(texts: list[str], source: str = "test.pdf") -> list[dict]:
    """Helper to create page dicts for testing."""
    return [
        {"text": text, "source": source, "page_number": i + 1}
        for i, text in enumerate(texts)
    ]


class TestChunkText:
    """Tests for chunk_text()."""

    def test_basic_chunking(self):
        """Chunks are created from a single page of text."""
        pages = make_pages(["A" * 500])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk

    def test_chunk_metadata_preserved(self):
        """Each chunk carries the correct source and page number."""
        pages = make_pages(["Some text about AI policy."], source="report.pdf")
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=100)

        assert chunks[0]["source"] == "report.pdf"
        assert chunks[0]["page_number"] == 1

    def test_chunk_indices_sequential(self):
        """Chunk indices are sequential starting from 0."""
        pages = make_pages(["Word " * 500, "More " * 500])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)

        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_multi_page_chunking(self):
        """Chunks span multiple pages correctly."""
        pages = make_pages(["Page one. " * 100, "Page two. " * 100])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)

        page_numbers = {c["page_number"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size produces exactly one chunk."""
        pages = make_pages(["Short text."])
        chunks = chunk_text(pages, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_empty_pages_no_chunks(self):
        """Empty input produces no chunks."""
        chunks = chunk_text([], chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_overlap_must_be_less_than_size(self):
        """Raises ValueError if overlap >= chunk_size."""
        pages = make_pages(["Some text."])

        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            chunk_text(pages, chunk_size=100, chunk_overlap=100)

        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            chunk_text(pages, chunk_size=100, chunk_overlap=200)

    def test_no_empty_chunks(self):
        """No chunk should have empty or whitespace-only text."""
        pages = make_pages(["Word " * 500])
        chunks = chunk_text(pages, chunk_size=200, chunk_overlap=50)

        for chunk in chunks:
            assert chunk["text"].strip() != ""
