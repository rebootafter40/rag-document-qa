"""
vector_store.py — Store and search document embeddings using ChromaDB.

Provides a simple interface to add document chunks and query
for the most similar ones.
"""
import logging

import chromadb
from pathlib import Path

logger = logging.getLogger(__name__)

# Store the database in the project directory
DB_PATH = "data/chroma_db"


def get_collection(collection_name: str = "documents") -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.

    Args:
        collection_name: Name of the collection.

    Returns:
        A ChromaDB collection ready for adding/querying documents.
    """
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    return collection


def add_documents(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Add document chunks and their embeddings to the vector store.

    Args:
        chunks: List of chunk dicts from chunker.py (must have 'text', 'source',
                'page_number', 'chunk_index').
        embeddings: List of embedding vectors matching the chunks.

    Raises:
        ValueError: If chunks and embeddings counts don't match,
            or if inputs are empty.
    """
    if not chunks:
        raise ValueError("No chunks to add. The document may be empty.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
        )

    collection = get_collection()

    # ChromaDB needs string IDs for each document
    ids = [f"{c['source']}_chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "page_number": c["page_number"]}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info("Added %d chunks to vector store", len(chunks))


def query(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the most similar chunks to a query embedding.

    Args:
        query_embedding: The embedding vector of the search query.
        top_k: Number of results to return.

    Returns:
        A list of dicts, each containing:
            - text (str): The chunk text
            - source (str): Source filename
            - page_number (int): Page number
            - distance (float): Cosine distance (lower = more similar)
        Returns an empty list if the collection has no documents.
    """
    collection = get_collection()

    # Handle empty collection — return early instead of erroring
    doc_count = collection.count()
    if doc_count == 0:
        return []

    # Don't request more results than exist
    n_results = min(top_k, doc_count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    # Unpack ChromaDB's nested result format
    output = []
    for i in range(len(results["ids"][0])):
        output.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
        })

    return output


def clear_collection(collection_name: str = "documents") -> None:
    """Delete a collection and all its data."""
    Path(DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(collection_name)
        logger.info("Cleared collection '%s'", collection_name)
    except Exception:
        logger.info("Collection '%s' does not exist", collection_name)

def delete_document(source_name: str) -> None:
    """Remove all chunks belonging to a specific document."""
    collection = get_collection()
    results = collection.get(where={"source": source_name})

    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info("Deleted %d chunks for '%s'", len(results["ids"]), source_name)
    else:
        logger.info("No chunks found for '%s'", source_name)

if __name__ == "__main__":
    from src.document_loader import load_pdf
    from src.chunker import chunk_text
    from src.embeddings import embed_texts, embed_text

    # Full pipeline test: load → chunk → embed → store → query
    print("=== Loading and processing PDF ===")
    pages = load_pdf("data/sample_docs/Americas-AI-Action-Plan.pdf")
    chunks = chunk_text(pages)

    print("\n=== Embedding chunks ===")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    print("\n=== Storing in vector database ===")
    clear_collection()  # Start fresh for testing
    add_documents(chunks, embeddings)

    print("\n=== Testing search ===")
    test_query = "What does the plan say about AI safety and security?"
    print(f"Query: {test_query}\n")

    query_vector = embed_text(test_query)
    results = query(query_vector, top_k=3)

    for i, result in enumerate(results):
        print(f"--- Result {i+1} (Page {result['page_number']}, Distance: {result['distance']:.4f}) ---")
        print(result["text"][:300])
        print("...\n")