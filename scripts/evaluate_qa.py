"""
evaluate_qa.py — End-to-end QA evaluation for the RAG pipeline.

Unlike scripts/evaluate_retrieval.py (which measures whether the right CHUNKS
are retrieved), this script measures the FULL pipeline: given a question, does
ask() return a correct answer that cites the right page(s)?

It runs a fixed test set against a known document (America's AI Action Plan),
auto-scores what can be scored objectively, and optionally prompts for a manual
correctness judgment on each answer. Results are written to a markdown table.

Scoring:
  - Citation accuracy (auto): did the cited pages include an expected page,
    within a +/-1 page tolerance? Chunks can span pages, and a chunk's recorded
    page number is where it STARTED, so the page holding the answer text may be
    one position later. The tolerance absorbs that.
  - Keyword coverage (auto): how many required keywords appear in the answer?
    A rough proxy for "did it talk about the right thing," not correctness.
  - Refusal (auto): for out-of-scope questions, did the system correctly decline
    instead of inventing an answer? Heuristic — eyeball these.
  - Correctness (manual): did it actually answer correctly? Prompted at runtime
    unless --auto is passed.

Usage:
    python scripts/evaluate_qa.py          # interactive: prompts for correctness
    python scripts/evaluate_qa.py --auto   # no prompts: leaves Correct as TODO

NOTE: This resets the vector store and ingests only the test document, so the
run is reproducible. Re-upload any other documents through the app afterward.
"""
import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Make `src` importable when this file is run directly (python scripts/evaluate_qa.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.qa_chain import ask                      # noqa: E402  (import after path tweak)
from src.retriever import ingest_pdf              # noqa: E402
from src.vector_store import clear_collection     # noqa: E402

# Quiet the noisy third-party loggers so the eval output stays readable.
for _noisy in ("httpx", "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# --- Configuration ---
TEST_DOCUMENT = "data/sample_docs/Americas-AI-Action-Plan.pdf"
RESULTS_PATH = "docs/eval_results.md"
PAGE_TOLERANCE = 1  # a cited page counts if within +/-1 of an expected page


@dataclass(frozen=True)
class TestCase:
    """A single evaluation question and its ground truth."""

    id: int
    question: str
    expected_pages: list[int]  # PDF-position page numbers; empty list = should refuse
    keywords: list[str]        # terms a correct answer should contain (matched as substrings)
    category: str              # "list" | "factual" | "synthesis" | "negative"

    @property
    def is_refusal(self) -> bool:
        """A question with no expected pages is an out-of-scope (refusal) test."""
        return len(self.expected_pages) == 0


# Ground truth verified against the actual PDF. Pages are PDF positions
# (PDF position = printed page + 3, since pages 1-3 are cover/epigraph/TOC).
# Some keywords are stems (e.g. "regulat") to match word variants like
# "regulatory"/"regulation" with a simple case-insensitive substring test.
TEST_SET: list[TestCase] = [
    TestCase(1, "What are the three pillars of America's AI Action Plan?",
             [4, 5], ["innovation", "infrastructure"], "list"),
    TestCase(2, "What does the plan say about open-source and open-weight AI models?",
             [7, 8], ["open", "models"], "factual"),
    TestCase(3, "How does the plan propose to train the AI workforce?",
             [9, 10, 20], ["workforce", "training"], "synthesis"),
    TestCase(4, "What does the plan say about AI infrastructure and energy?",
             [17, 18, 19], ["infrastructure", "energy"], "factual"),
    TestCase(5, "How does the plan address competition with China?",
             [4, 17, 23], ["race", "dominance"], "synthesis"),
    TestCase(6, "What does the plan say about AI and national security?",
             [9, 13, 24], ["security"], "factual"),
    TestCase(7, "What is Pillar III about?",
             [23], ["international", "diplomacy"], "factual"),
    TestCase(8, "What does the plan recommend about regulatory barriers to AI?",
             [6, 7], ["regulat"], "factual"),
    TestCase(9, "What role does the federal government play in supporting open models?",
             [7, 8], ["federal", "open"], "factual"),
    TestCase(10, "How does the plan support displaced or retrained workers?",
             [10], ["retrain", "displac"], "factual"),
    TestCase(11, "What is the capital of France?",
             [], [], "negative"),
    TestCase(12, "What's a good recipe for chocolate chip cookies?",
             [], [], "negative"),
]


# --- Scoring helpers ---
def score_citation(
    cited_pages: list[int],
    expected_pages: list[int],
    tolerance: int = PAGE_TOLERANCE,
) -> bool:
    """
    Return True if any cited page is within `tolerance` of any expected page.

    The tolerance absorbs chunk-spanning: a chunk's recorded page number is
    where it started, so the page that actually contains the answer text may
    be one position later.
    """
    return any(
        abs(cited - expected) <= tolerance
        for cited in cited_pages
        for expected in expected_pages
    )


def score_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    """Return (found, total): how many keywords appear in the answer (case-insensitive)."""
    if not keywords:
        return (0, 0)
    low = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in low)
    return (found, len(keywords))


# Phrases that signal the system declined to answer from the document.
REFUSAL_MARKERS = (
    "no relevant information",
    "does not contain",
    "doesn't contain",
    "do not contain",
    "not contain",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "context does not",
    "context doesn't",
    "not in the provided context",
    "not provided in the context",
    "no information",
    "not mentioned",
    "does not provide",
    "doesn't provide",
    "not found in",
    "is not covered",
    "isn't covered",
)


def is_refusal(answer: str) -> bool:
    """
    Heuristic: did the system decline to answer from the document?

    Scans for grounded-decline phrasing. Imperfect by nature — always eyeball
    the negative tests in the console output rather than trusting this blindly.
    """
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


def prompt_correctness() -> bool | None:
    """Ask the human whether the answer was correct. Returns True / False / None (skip)."""
    while True:
        choice = input("Was this answer correct? [y]es / [n]o / [s]kip: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        if choice in ("s", "skip", ""):
            return None
        print("  Please enter y, n, or s.")


# --- Evaluation run ---
def run_evaluation(interactive: bool) -> list[dict]:
    """Ingest the test document, run every test case, and score the results."""
    print(f"Resetting vector store and ingesting: {TEST_DOCUMENT}")
    print("(This clears the current collection — expected for a clean eval run.)\n")
    clear_collection()
    num_chunks = ingest_pdf(TEST_DOCUMENT)
    print(f"Ingested {num_chunks} chunks.")
    print("=" * 70)

    results: list[dict] = []

    for case in TEST_SET:
        print(f"\n[{case.id}/{len(TEST_SET)}] {case.question}")
        print("-" * 70)

        response = ask(case.question)
        answer = response["answer"]
        cited_pages = sorted({s["page_number"] for s in response["sources"]})

        print(f"{answer}\n")
        print(f"Cited pages: {cited_pages or '(none)'}")

        row: dict = {
            "id": case.id,
            "question": case.question,
            "category": case.category,
            "cited_pages": cited_pages,
            "expected_pages": case.expected_pages,
        }

        if case.is_refusal:
            refused = is_refusal(answer)
            row["refusal_ok"] = refused
            print(f"Expected: REFUSE   |   Refused: {'YES' if refused else 'NO'}")
        else:
            cite_ok = score_citation(cited_pages, case.expected_pages)
            found, total = score_keywords(answer, case.keywords)
            row["citation_ok"] = cite_ok
            row["keywords_found"] = found
            row["keywords_total"] = total
            print(
                f"Expected pages: {case.expected_pages}   |   "
                f"Citation: {'PASS' if cite_ok else 'FAIL'}   |   "
                f"Keywords: {found}/{total}"
            )

        if interactive:
            row["correct"] = prompt_correctness()
        else:
            row["correct"] = None

        results.append(row)

    return results


# --- Reporting ---
def _mark(value: bool | None) -> str:
    """Render a tri-state (pass / fail / unknown) as a markdown table cell."""
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "TODO"


def _pct(num: int, denom: int) -> str:
    return f"{(100 * num / denom):.0f}%" if denom else "n/a"


def write_markdown_report(results: list[dict], path: str) -> None:
    """Write a markdown results table plus summary for the README / docs."""
    lines = [
        "# End-to-End QA Evaluation Results",
        "",
        f"_Document:_ `{TEST_DOCUMENT}`  ",
        f"_Run:_ {datetime.now():%Y-%m-%d %H:%M}  ",
        f"_Page tolerance:_ ±{PAGE_TOLERANCE}",
        "",
        "| # | Question | Type | Cited | Expected | Citation | Keywords | Refusal | Correct |",
        "|---|----------|------|-------|----------|----------|----------|---------|---------|",
    ]

    for r in results:
        cited = ", ".join(map(str, r["cited_pages"])) or "—"
        if r["category"] == "negative":
            expected, citation, keywords = "refuse", "—", "—"
            refusal = _mark(r.get("refusal_ok"))
        else:
            expected = ", ".join(map(str, r["expected_pages"]))
            citation = _mark(r.get("citation_ok"))
            keywords = f"{r['keywords_found']}/{r['keywords_total']}"
            refusal = "—"
        correct = _mark(r.get("correct"))
        question = r["question"] if len(r["question"]) <= 60 else r["question"][:57] + "..."
        lines.append(
            f"| {r['id']} | {question} | {r['category']} | {cited} | {expected} | "
            f"{citation} | {keywords} | {refusal} | {correct} |"
        )

    # Aggregate metrics
    answerable = [r for r in results if r["category"] != "negative"]
    negatives = [r for r in results if r["category"] == "negative"]
    cite_hits = sum(1 for r in answerable if r.get("citation_ok"))
    kw_found = sum(r.get("keywords_found", 0) for r in answerable)
    kw_total = sum(r.get("keywords_total", 0) for r in answerable)
    refusal_hits = sum(1 for r in negatives if r.get("refusal_ok"))
    scored = [r for r in results if r.get("correct") is not None]
    correct_hits = sum(1 for r in scored if r["correct"])

    lines += [
        "",
        "## Summary",
        "",
        f"- **Citation accuracy:** {cite_hits}/{len(answerable)} ({_pct(cite_hits, len(answerable))})",
        f"- **Keyword coverage:** {kw_found}/{kw_total} ({_pct(kw_found, kw_total)})",
        f"- **Refusal accuracy:** {refusal_hits}/{len(negatives)} ({_pct(refusal_hits, len(negatives))})",
    ]
    if scored:
        lines.append(f"- **Manual correctness:** {correct_hits}/{len(scored)} ({_pct(correct_hits, len(scored))})")
    else:
        lines.append("- **Manual correctness:** _not yet scored — fill in the Correct column above_")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults written to {path}")


def print_summary(results: list[dict]) -> None:
    """Print a compact summary to the console."""
    answerable = [r for r in results if r["category"] != "negative"]
    negatives = [r for r in results if r["category"] == "negative"]
    cite_hits = sum(1 for r in answerable if r.get("citation_ok"))
    refusal_hits = sum(1 for r in negatives if r.get("refusal_ok"))
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Citation accuracy: {cite_hits}/{len(answerable)}")
    print(f"  Refusal accuracy:  {refusal_hits}/{len(negatives)}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end QA evaluation for the RAG pipeline."
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Skip the manual correctness prompts; leave Correct as TODO in the report.",
    )
    args = parser.parse_args()

    results = run_evaluation(interactive=not args.auto)
    print_summary(results)
    write_markdown_report(results, RESULTS_PATH)


if __name__ == "__main__":
    main()