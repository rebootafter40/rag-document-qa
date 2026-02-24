"""
document_loader.py — Extract text from PDF documents.

Uses PyMuPDF (fitz) to read PDF files and return structured text
with page-level metadata.
"""

import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one entry per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts, each containing:
            - page_number (int): 1-indexed page number
            - text (str): Extracted text from that page
            - source (str): Filename of the PDF
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip pages with little or no text (e.g., cover images)
        if text.strip():
            pages.append({
                "page_number": page_num + 1,  # 1-indexed for humans
                "text": text.strip(),
                "source": path.name,
            })

    doc.close()

    print(f"Loaded {len(pages)} pages from '{path.name}'")
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