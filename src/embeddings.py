"""
embeddings.py — Convert text into vector embeddings.
Uses sentence-transformers to generate embeddings locally (free, no API needed).
"""
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Load the model once when this module is imported
# all-MiniLM-L6-v2 is small, fast, and good quality for semantic search
_model = None


def get_model() -> SentenceTransformer:
    """
    Load and cache the embedding model.

    Uses lazy loading so the model is only downloaded/loaded
    the first time it's needed.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model (first time may download ~80MB)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded.")
    return _model


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single piece of text.

    Args:
        text: The string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts at once (more efficient).

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()


if __name__ == "__main__":
    # Quick test: embed a few sample sentences and check similarity
    test_sentences = [
        "AI policy and regulation in the United States",
        "Artificial intelligence rules and laws in America",
        "How to bake chocolate chip cookies",
    ]

    print("Embedding test sentences...")
    vectors = embed_texts(test_sentences)

    print(f"\nVector dimension: {len(vectors[0])}")
    print(f"Number of vectors: {len(vectors)}")

    # Simple cosine similarity to show it works
    from numpy import dot
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    print(f"\nSimilarity: 'AI policy' vs 'AI rules': {cosine_sim(vectors[0], vectors[1]):.4f}")
    print(f"Similarity: 'AI policy' vs 'cookies':   {cosine_sim(vectors[0], vectors[2]):.4f}")