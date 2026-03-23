"""Tests for the QA chain module."""
import pytest
from src.qa_chain import build_context


class TestBuildContext:
    """Tests for the context formatting function."""

    def test_single_result(self):
        """Formats a single result correctly."""
        results = [
            {
                "text": "AI is transforming industries.",
                "source": "report.pdf",
                "page_number": 5,
                "distance": 0.2,
            }
        ]

        context = build_context(results)

        assert "Source 1" in context
        assert "report.pdf" in context
        assert "Page 5" in context
        assert "AI is transforming industries." in context

    def test_multiple_results(self):
        """Formats multiple results with numbered sources."""
        results = [
            {
                "text": "First chunk.",
                "source": "doc.pdf",
                "page_number": 1,
                "distance": 0.1,
            },
            {
                "text": "Second chunk.",
                "source": "doc.pdf",
                "page_number": 3,
                "distance": 0.3,
            },
        ]

        context = build_context(results)

        assert "Source 1" in context
        assert "Source 2" in context
        assert "First chunk." in context
        assert "Second chunk." in context

    def test_empty_results(self):
        """Empty results produce empty context."""
        context = build_context([])
        assert context == ""

    def test_results_separated_by_divider(self):
        """Results are separated by a divider for clarity."""
        results = [
            {
                "text": "Chunk A.",
                "source": "a.pdf",
                "page_number": 1,
                "distance": 0.1,
            },
            {
                "text": "Chunk B.",
                "source": "b.pdf",
                "page_number": 2,
                "distance": 0.2,
            },
        ]

        context = build_context(results)

        assert "---" in context