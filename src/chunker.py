"""
chunker.py — Split extracted text into overlapping chunks.
Chunks carry metadata (source file, page number, chunk index)
so that answers can cite their sources later.
"""

import logging

from src.config import settings

logger = logging.getLogger(__name__)


def chunk_text(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Split page text into overlapping chunks.

    Args:
        pages: Output from document_loader.load_pdf().
        chunk_size: Target size of each chunk in characters.
            Defaults to settings.chunk_size when not provided.
        chunk_overlap: Number of characters to overlap between chunks.
            Defaults to settings.chunk_overlap when not provided.

    Returns:
        A list of dicts, each containing:
            - text (str): The chunk text
            - source (str): Filename of the PDF
            - page_number (int): Page the chunk started on
            - chunk_index (int): Global index of this chunk
    """
    # Fall back to configured defaults when the caller doesn't override them.
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        source = page["source"]
        page_number = page["page_number"]

        # Slide a window across the text
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Get the chunk text
            chunk_text_content = text[start:end]

            # Try to break at a natural boundary (newline or sentence end)
            if end < len(text):
                # Look for last newline in the chunk
                last_newline = chunk_text_content.rfind("\n")
                last_period = chunk_text_content.rfind(". ")

                # Pick the best break point (prefer newline, then period)
                break_point = max(last_newline, last_period)

                if break_point > chunk_size * settings.chunk_break_threshold:
                    # Only use it if it's past the configured threshold
                    chunk_text_content = text[start : start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(
                {
                    "text": chunk_text_content.strip(),
                    "source": source,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

            # Move forward by (end - overlap), so chunks overlap
            start = end - chunk_overlap

    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


if __name__ == "__main__":
    import sys
    from src.document_loader import load_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m src.chunker <path-to-pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    chunks = chunk_text(pages)

    # Show a few chunks so we can inspect quality
    for chunk in chunks[:3]:
        print(f"\n--- Chunk {chunk['chunk_index']} (Page {chunk['page_number']}) ---")
        print(f"Length: {len(chunk['text'])} chars")
        print(chunk["text"][:300])
        print("...")
