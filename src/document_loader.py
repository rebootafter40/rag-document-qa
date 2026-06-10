"""
document_loader.py — Extract text from PDF documents.
Uses PyMuPDF (fitz) to read PDF files and return structured text
with page-level metadata.
"""

import logging

import fitz  # PyMuPDF
from pathlib import Path

logger = logging.getLogger(__name__)

# Limits to prevent resource issues with very large documents
MAX_FILE_SIZE_MB = 50
MAX_PAGES = 200


def load_pdf(file_path: str, original_filename: str | None = None) -> list[dict]:
    """
    Extract text from a PDF file, one entry per page.

    Args:
        file_path: Path to the PDF file.
        original_filename: Display name for the source file. If None, uses the
            actual filename from file_path. Useful when file_path is a temp file.

    Returns:
        A list of dicts, each containing:
            - page_number (int): 1-indexed page number
            - text (str): Extracted text from that page
            - source (str): Filename of the PDF

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a PDF, exceeds size/page limits,
            is corrupt, or contains no extractable text.
    """
    path = Path(file_path)
    source_name = original_filename or path.name

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    # Check file size before opening
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"'{source_name}' is too large ({file_size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    # Attempt to open the PDF (catches corrupt/malformed files)
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise ValueError(
            f"Could not open '{source_name}'. The file may be corrupt "
            f"or password-protected. Details: {e}"
        ) from e

    # Check page count
    if len(doc) > MAX_PAGES:
        doc.close()
        raise ValueError(
            f"'{source_name}' has {len(doc)} pages. "
            f"Maximum allowed is {MAX_PAGES} pages."
        )

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip pages with little or no text (e.g., cover images)
        if text.strip():
            pages.append(
                {
                    "page_number": page_num + 1,  # 1-indexed for humans
                    "text": text.strip(),
                    "source": source_name,
                }
            )

    doc.close()

    # Catch image-based PDFs with no extractable text
    if not pages:
        raise ValueError(
            f"No readable text found in '{source_name}'. "
            "The PDF may be image-based (scanned). "
            "This app requires text-based PDFs."
        )

    logger.info("Loaded %d pages from '%s'", len(pages), source_name)
    return pages


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.document_loader <path-to-pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])

    # Preview first 3 pages
    for page in pages[:3]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:500])  # First 500 chars
        print("...")
