"""
qa_chain.py — Generate answers using Claude and retrieved context.
This is the core RAG function: take a question, retrieve relevant
chunks, and ask Claude to answer based only on that context.
"""
from anthropic import (
    Anthropic,
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

from src.config import settings
from src.retriever import retrieve

# API key now comes from the central settings object (loaded from .env by
# pydantic-settings), so no separate load_dotenv()/os.getenv() needed here.
client = Anthropic(api_key=settings.anthropic_api_key)

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


def ask(
    question: str,
    top_k: int | None = None,
    use_reranking: bool | None = None,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Answer a question using RAG: retrieve context, then ask Claude.

    Args:
        question: The user's natural language question.
        top_k: Number of chunks to retrieve for context. Defaults to
            settings.top_k (resolved in retrieve()).
        use_reranking: If True, use cross-encoder reranking. Defaults to
            settings.use_reranking_default (resolved in retrieve()).
        conversation_history: Optional list of prior message dicts with
            'role' and 'content' keys. Used for follow-up questions.

    Returns:
        A dict containing:
            - answer (str): Claude's response
            - sources (list[dict]): The retrieved chunks used as context

    Raises:
        ValueError: If the question is empty or too long.
        RuntimeError: If the API call fails (auth, rate limit, timeout, etc.).
    """
    # Validate input
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if len(question) > settings.max_question_length:
        raise ValueError(
            f"Question is too long ({len(question)} characters). "
            f"Please keep questions under {settings.max_question_length} characters."
        )

    # Check for API key early so the error message is clear
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Anthropic API key not found. "
            "Make sure ANTHROPIC_API_KEY is set in your .env file."
        )

    # Step 1: Retrieve relevant chunks (top_k / reranking resolve from config)
    results = retrieve(question, top_k=top_k, use_reranking=use_reranking)

    # Step 2: Handle case where no relevant chunks are found
    if not results:
        return {
            "answer": (
                "No relevant information found in the uploaded documents. "
                "Try rephrasing your question or uploading a different document."
            ),
            "sources": [],
        }

    # Step 3: Build the context from retrieved chunks
    context = build_context(results)

    # Step 4: Create the prompt with context + question
    user_message = f"""Context from documents:

{context}

---

Question: {question}

Please answer based only on the context provided above. Cite the source
and page number for any claims you make."""

    # Build messages with conversation history for follow-up questions
    messages = []
    if conversation_history:
        # Keep the last N exchanges (each exchange = 1 user + 1 assistant msg)
        # to give Claude follow-up context without bloating token usage.
        recent_history = conversation_history[-(settings.history_exchanges * 2):]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    # Step 5: Call Claude with error handling
    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except AuthenticationError:
        raise RuntimeError(
            "Invalid API key. Please check your ANTHROPIC_API_KEY in the .env file."
        )
    except RateLimitError:
        raise RuntimeError(
            "Rate limit exceeded. Please wait a moment and try again."
        )
    except APITimeoutError:
        raise RuntimeError(
            "The request to Claude timed out. Please try again."
        )
    except APIConnectionError:
        raise RuntimeError(
            "Could not connect to the Anthropic API. "
            "Please check your internet connection."
        )
    except APIStatusError as e:
        raise RuntimeError(f"Anthropic API error: {e.message}") from e

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