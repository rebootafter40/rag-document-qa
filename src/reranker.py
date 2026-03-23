"""
reranker.py — Rerank retrieved chunks using a cross-encoder model.

Cross-encoders are more accurate than bi-encoders (embeddings) for
relevance scoring because they process the query and document together.
The tradeoff is speed — so we use the bi-encoder for initial retrieval
(fast, over all chunks) and the cross-encoder to rerank just the top
candidates (slow but accurate, over a small set).
"""
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Small, fast cross-encoder fine-tuned for passage reranking
_reranker = None


def get_reranker() -> CrossEncoder:
    """
    Load and cache the cross-encoder reranking model.

    Uses lazy loading so the model is only downloaded/loaded
    the first time it's needed.
    """
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model (first time may download ~80MB)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker model loaded.")
    return _reranker


def rerank(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks by relevance to the query.

    Takes the initial retrieval results (from bi-encoder/vector search)
    and rescores them using a cross-encoder for more accurate ranking.

    Args:
        query: The user's question.
        results: List of result dicts from retriever.retrieve().
            Each must have a 'text' key.
        top_k: Number of top results to return after reranking.

    Returns:
        The top_k results sorted by cross-encoder relevance score
        (highest first). Each dict gets an added 'rerank_score' field.
    """
    if not results:
        return []

    model = get_reranker()

    # Cross-encoder takes (query, passage) pairs
    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    # Attach scores to results
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    # Sort by cross-encoder score (highest = most relevant)
    reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)

    return reranked[:top_k]


if __name__ == "__main__":
    # Quick test: rerank some sample results
    sample_results = [
        {"text": "The plan focuses on building new data centers and energy infrastructure."},
        {"text": "Workers will need retraining as AI changes the job market."},
        {"text": "America must lead in AI chip manufacturing and semiconductor production."},
    ]

    query_text = "What does the plan say about semiconductor manufacturing?"
    print(f"Query: {query_text}\n")

    reranked = rerank(query_text, sample_results)

    for i, r in enumerate(reranked, 1):
        print(f"{i}. (score: {r['rerank_score']:.4f}) {r['text']}")