"""
retriever.py — Clean interface for document retrieval.

Combines embedding and vector search into a single
retrieve() function that the QA chain will use.
"""

from src.embeddings import embed_text, embed_texts
from src.vector_store import add_documents, query, clear_collection
from src.document_loader import load_pdf
from src.chunker import chunk_text


def ingest_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    """
    Process a PDF file and add it to the vector store.

    This runs the full ingestion pipeline:
    load PDF → chunk text → generate embeddings → store in vector DB.

    Args:
        file_path: Path to the PDF file.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        The number of chunks created and stored.
    """
    pages = load_pdf(file_path)
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    add_documents(chunks, embeddings)

    return len(chunks)


def retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Find the most relevant document chunks for a question.

    Args:
        query_text: The user's question.
        top_k: Number of chunks to return.

    Returns:
        A list of dicts with 'text', 'source', 'page_number', and 'distance'.
    """
    query_vector = embed_text(query_text)
    results = query(query_vector, top_k=top_k)
    return results


if __name__ == "__main__":
    # Full test: ingest a document, then ask questions
    print("=== Ingesting PDF ===")
    clear_collection()
    num_chunks = ingest_pdf("data/sample_docs/Americas-AI-Action-Plan.pdf")
    print(f"Stored {num_chunks} chunks\n")

    # Test with a few different questions
    test_questions = [
        "What does the plan say about AI workforce training?",
        "How will the US compete with China on AI?",
        "What is the plan for AI infrastructure and energy?",
    ]

    for question in test_questions:
        print(f"Q: {question}")
        results = retrieve(question, top_k=2)
        for r in results:
            print(f"  → Page {r['page_number']} (distance: {r['distance']:.4f}): {r['text'][:150]}...")
        print()