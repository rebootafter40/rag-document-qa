"""
qa_chain.py — Generate answers using Claude and retrieved context.

This is the core RAG function: take a question, retrieve relevant
chunks, and ask Claude to answer based only on that context.
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic
from src.retriever import retrieve

# Load API key from .env file
load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a helpful document Q&A assistant. Your job is to answer
questions based ONLY on the provided context from the user's documents.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context doesn't contain enough information to answer, say so honestly.
   Don't make things up.
3. When you reference information, cite the source and page number
   (e.g., "According to [source], page X...").
4. Keep answers clear and concise.
5. If the question is ambiguous, state your interpretation before answering."""


def build_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    Args:
        results: List of result dicts from retriever.retrieve().

    Returns:
        A formatted string with all chunks and their metadata.
    """
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[Source {i}: {r['source']}, Page {r['page_number']}]\n"
            f"{r['text']}\n"
        )
    return "\n---\n".join(context_parts)


def ask(question: str, top_k: int = 5) -> dict:
    """
    Answer a question using RAG: retrieve context, then ask Claude.

    Args:
        question: The user's natural language question.
        top_k: Number of chunks to retrieve for context.

    Returns:
        A dict containing:
            - answer (str): Claude's response
            - sources (list[dict]): The retrieved chunks used as context
    """
    # Step 1: Retrieve relevant chunks
    results = retrieve(question, top_k=top_k)

    # Step 2: Build the context from retrieved chunks
    context = build_context(results)

    # Step 3: Create the prompt with context + question
    user_message = f"""Context from documents:

{context}

---

Question: {question}

Please answer based only on the context provided above. Cite the source
and page number for any claims you make."""

    # Step 4: Call Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    answer = response.content[0].text

    return {
        "answer": answer,
        "sources": results,
    }


if __name__ == "__main__":
    from src.retriever import ingest_pdf
    from src.vector_store import clear_collection

    # First, make sure the document is ingested
    print("=== Setting up vector store ===")
    clear_collection()
    ingest_pdf("data/sample_docs/Americas-AI-Action-Plan.pdf")

    # Test questions
    test_questions = [
        "What are the three pillars of America's AI Action Plan?",
        "What does the plan say about open-source AI?",
        "How does the plan address AI workforce training?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")

        result = ask(question)

        print(f"\nA: {result['answer']}")

        print(f"\nSources used:")
        for s in result["sources"]:
            print(f"  - {s['source']}, Page {s['page_number']} (distance: {s['distance']:.4f})")