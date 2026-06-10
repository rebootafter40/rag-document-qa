"""
evaluate_retrieval.py — Compare retrieval quality across different configurations.

Tests multiple chunk size settings, with and without reranking,
against a set of questions with known answer locations.
Outputs results as a formatted table for documentation.
"""

import sys
import time
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import load_pdf
from src.chunker import chunk_text
from src.embeddings import embed_text, embed_texts
from src.vector_store import add_documents, clear_collection, query
from src.reranker import rerank
from src.logging_config import setup_logging

setup_logging()

# --- Test Set ---
# Each question has:
#   - question: the natural language query
#   - expected_pages: pages where the answer is known to exist
#   - key_terms: terms that should appear in relevant chunks
TEST_SET = [
    {
        "question": "What are the three pillars of the AI Action Plan?",
        "expected_pages": [4],
        "key_terms": ["innovation", "infrastructure", "diplomacy"],
    },
    {
        "question": "What does the plan say about open-source AI?",
        "expected_pages": [7, 8],
        "key_terms": ["open-source", "open-weight", "startups"],
    },
    {
        "question": "How will AI affect American workers?",
        "expected_pages": [9, 10],
        "key_terms": ["workforce", "worker", "retraining", "labor"],
    },
    {
        "question": "What is the plan for semiconductor manufacturing?",
        "expected_pages": [19],
        "key_terms": ["semiconductor", "chips", "manufacturing"],
    },
    {
        "question": "How does the plan address AI safety and security risks?",
        "expected_pages": [12, 21, 22],
        "key_terms": ["interpretability", "robustness", "security", "vulnerabilities"],
    },
    {
        "question": "What role does the Department of Defense play in AI?",
        "expected_pages": [14, 15],
        "key_terms": ["defense", "DOD", "military", "warfighting"],
    },
    {
        "question": "What does the plan say about AI and energy infrastructure?",
        "expected_pages": [17, 18],
        "key_terms": ["energy", "grid", "power", "data center"],
    },
    {
        "question": "How will the US counter China's AI influence?",
        "expected_pages": [23, 24],
        "key_terms": ["china", "chinese", "export control", "adversar"],
    },
    {
        "question": "What is the plan for AI in scientific research?",
        "expected_pages": [11],
        "key_terms": ["science", "research", "dataset", "lab"],
    },
    {
        "question": "How does the plan address deepfakes and synthetic media?",
        "expected_pages": [15, 16],
        "key_terms": ["deepfake", "synthetic", "forensic", "evidence"],
    },
]

# Chunk configurations to test: (chunk_size, chunk_overlap)
CONFIGS = [
    (500, 100),
    (1000, 200),
    (1500, 300),
]

PDF_PATH = "data/sample_docs/Americas-AI-Action-Plan.pdf"


def score_results(
    results: list[dict], expected_pages: list[int], key_terms: list[str]
) -> dict:
    """
    Score a set of retrieval results against expected answers.

    Returns:
        Dict with:
            - page_hit (bool): Did any result come from an expected page?
            - term_hits (int): How many key terms appeared in results?
            - term_total (int): Total key terms expected.
            - term_ratio (float): Fraction of key terms found.
    """
    retrieved_pages = [r["page_number"] for r in results]
    combined_text = " ".join(r["text"].lower() for r in results)

    page_hit = any(p in retrieved_pages for p in expected_pages)
    term_hits = sum(1 for t in key_terms if t.lower() in combined_text)

    return {
        "page_hit": page_hit,
        "term_hits": term_hits,
        "term_total": len(key_terms),
        "term_ratio": term_hits / len(key_terms) if key_terms else 0,
    }


def run_evaluation(
    chunk_size: int, chunk_overlap: int, use_reranking: bool, top_k: int = 5
) -> dict:
    """
    Run the full evaluation for one configuration.

    Returns:
        Dict with aggregate scores.
    """
    # Ingest the document with this config
    clear_collection()
    pages = load_pdf(PDF_PATH)
    chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    add_documents(chunks, embeddings)

    num_chunks = len(chunks)
    page_hits = 0
    total_term_ratio = 0

    for test in TEST_SET:
        # Retrieve
        query_vec = embed_text(test["question"])
        # Retrieve more candidates if reranking, then narrow down
        retrieve_k = top_k * 2 if use_reranking else top_k
        results = query(query_vec, top_k=retrieve_k)

        if use_reranking:
            results = rerank(test["question"], results, top_k=top_k)

        # Score
        scores = score_results(results, test["expected_pages"], test["key_terms"])
        if scores["page_hit"]:
            page_hits += 1
        total_term_ratio += scores["term_ratio"]

    num_questions = len(TEST_SET)
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": num_chunks,
        "reranking": use_reranking,
        "page_hit_rate": page_hits / num_questions,
        "avg_term_coverage": total_term_ratio / num_questions,
    }


def main():
    print("=" * 70)
    print("RAG Retrieval Quality Evaluation")
    print("=" * 70)
    print(f"Document: {PDF_PATH}")
    print(f"Test questions: {len(TEST_SET)}")
    print(f"Configurations: {len(CONFIGS)} chunk sizes × 2 (with/without reranking)")
    print()

    all_results = []

    for chunk_size, chunk_overlap in CONFIGS:
        for use_reranking in [False, True]:
            label = f"chunks={chunk_size}/{chunk_overlap}, rerank={'yes' if use_reranking else 'no'}"
            print(f"Testing: {label}...")
            start = time.time()

            result = run_evaluation(chunk_size, chunk_overlap, use_reranking)
            elapsed = time.time() - start

            result["time_seconds"] = round(elapsed, 1)
            all_results.append(result)

            print(
                f"  → {result['num_chunks']} chunks, "
                f"page hits: {result['page_hit_rate']:.0%}, "
                f"term coverage: {result['avg_term_coverage']:.0%}, "
                f"time: {elapsed:.1f}s"
            )

    # Print summary table
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'Chunks':>6} {'Pages':>7} {'Terms':>7} {'Time':>6}")
    print("-" * 70)

    for r in all_results:
        rerank_label = "+rerank" if r["reranking"] else "       "
        config = f"{r['chunk_size']}/{r['chunk_overlap']} {rerank_label}"
        print(
            f"{config:<30} {r['num_chunks']:>6} "
            f"{r['page_hit_rate']:>6.0%} "
            f"{r['avg_term_coverage']:>6.0%} "
            f"{r['time_seconds']:>5.1f}s"
        )

    # Find best config
    best = max(all_results, key=lambda r: (r["page_hit_rate"], r["avg_term_coverage"]))
    rerank_str = " with reranking" if best["reranking"] else ""
    print()
    print(
        f"Best config: chunk_size={best['chunk_size']}, "
        f"overlap={best['chunk_overlap']}{rerank_str}"
    )
    print(f"  Page hit rate: {best['page_hit_rate']:.0%}")
    print(f"  Term coverage: {best['avg_term_coverage']:.0%}")


if __name__ == "__main__":
    main()
