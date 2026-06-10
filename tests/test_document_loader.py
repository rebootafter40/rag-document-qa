"""Tests for the PDF document loader module."""

import pytest
from pathlib import Path
from src.document_loader import load_pdf


class TestLoadPdfValidation:
    """Tests for input validation in load_pdf()."""

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for nonexistent files."""
        fake_path = str(tmp_path / "nonexistent.pdf")

        with pytest.raises(FileNotFoundError):
            load_pdf(fake_path)

    def test_wrong_file_extension(self, tmp_path):
        """Raises ValueError for non-PDF files."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Hello")

        with pytest.raises(ValueError, match="Expected a PDF"):
            load_pdf(str(txt_file))

    def test_corrupt_pdf(self, tmp_path):
        """Raises ValueError for corrupt/invalid PDF files."""
        bad_pdf = tmp_path / "corrupt.pdf"
        bad_pdf.write_text("this is not a real pdf")

        with pytest.raises(ValueError, match="Could not open"):
            load_pdf(str(bad_pdf))

    def test_original_filename_override(self, tmp_path):
        """When original_filename is provided, it's used as the source."""
        # We can't easily create a real PDF in a test, so we test
        # the validation path — the filename appears in the error message
        bad_pdf = tmp_path / "tmp123.pdf"
        bad_pdf.write_text("not a pdf")

        with pytest.raises(ValueError, match="my_report.pdf"):
            load_pdf(str(bad_pdf), original_filename="my_report.pdf")


class TestLoadPdfWithRealPdf:
    """Tests that require the sample PDF to be present."""

    SAMPLE_PDF = "data/sample_docs/Americas-AI-Action-Plan.pdf"

    @pytest.fixture(autouse=True)
    def check_sample_exists(self):
        """Skip these tests if the sample PDF isn't available."""
        if not Path(self.SAMPLE_PDF).exists():
            pytest.skip("Sample PDF not found — skipping integration test")

    def test_loads_pages(self):
        """Successfully loads pages from a real PDF."""
        pages = load_pdf(self.SAMPLE_PDF)

        assert len(pages) > 0
        assert all("text" in p for p in pages)
        assert all("page_number" in p for p in pages)
        assert all("source" in p for p in pages)

    def test_pages_have_text(self):
        """Each loaded page has non-empty text."""
        pages = load_pdf(self.SAMPLE_PDF)

        for page in pages:
            assert page["text"].strip() != ""

    def test_page_numbers_are_sequential(self):
        """Page numbers start at 1 and increase."""
        pages = load_pdf(self.SAMPLE_PDF)

        page_nums = [p["page_number"] for p in pages]
        assert page_nums[0] >= 1
        assert page_nums == sorted(page_nums)

    def test_source_is_filename(self):
        """Source metadata is the PDF filename, not the full path."""
        pages = load_pdf(self.SAMPLE_PDF)

        assert pages[0]["source"] == "Americas-AI-Action-Plan.pdf"
