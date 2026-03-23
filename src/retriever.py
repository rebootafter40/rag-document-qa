"""
retriever.py — Clean interface for document retrieval.
Combines embedding and vector search into a single
retrieve() function that the QA chain will use.
"""
from src.embeddings import embed_text, embed_texts
from src.vector_store import add_documents, query, clear_collection
from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.reranker import rerank


def ingest_pdf(
    file_path: str,
    original_filename: str | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 300,
) -> int:
    """
    Process a PDF file and add it to the vector store.

    This runs the full ingestion pipeline:
    load PDF → chunk text → generate embeddings → store in vector DB.

    Args:
        file_path: Path to the PDF file.
        original_filename: Display name for the source file. If None, uses the
            actual filename from file_path. Useful when file_path is a temp file.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        The number of chunks created and stored.
    """
    pages = load_pdf(file_path, original_filename=original_filename)
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    add_documents(chunks, embeddings)

    return len(chunks)


def retrieve(
    query_text: str,
    top_k: int = 5,
    use_reranking: bool = False,
) -> list[dict]:
    """
    Find the most relevant document chunks for a question.

    Args:
        query_text: The user's question.
        top_k: Number of chunks to return.
        use_reranking: If True, retrieve extra candidates and rerank
            with a cross-encoder for better accuracy.

    Returns:
        A list of dicts with 'text', 'source', 'page_number', and 'distance'.
        If reranking is enabled, each dict also has 'rerank_score'.
    """
    query_vector = embed_text(query_text)

    if use_reranking:
        # Retrieve more candidates, then let the reranker pick the best
        candidates = query(query_vector, top_k=top_k * 2)
        results = rerank(query_text, candidates, top_k=top_k)
    else:
        results = query(query_vector, top_k=top_k)

    return results


if __name__ == "__main__":
    # Full test: ingest a document, then ask questions
    print("=== Ingesting PDF ===")
    clear_collection()
    num_chunks = ingest_pdf("data/sample_docs/Americas-AI-Action-Plan.pdf")
    print(f"Stored {num_chunks} chunks\n")

    # Test with a few different questions — compare with and without reranking
    test_questions = [
        "What does the plan say about AI workforce training?",
        "How will the US compete with China on AI?",
        "What is the plan for AI infrastructure and energy?",
    ]

    for question in test_questions:
        print(f"Q: {question}")

        print("  Without reranking:")
        results = retrieve(question, top_k=3, use_reranking=False)
        for r in results:
            print(f"    → Page {r['page_number']} (distance: {r['distance']:.4f}): {r['text'][:100]}...")

        print("  With reranking:")
        results = retrieve(question, top_k=3, use_reranking=True)
        for r in results:
            print(f"    → Page {r['page_number']} (rerank: {r['rerank_score']:.4f}): {r['text'][:100]}...")

        print()