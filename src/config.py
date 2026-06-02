"""
config.py — Central configuration for the RAG Document Q&A app.

All tunable values live here in a single, validated Settings object.
Any value can be overridden by an environment variable or a .env entry,
which makes deployment (e.g. Streamlit Cloud secrets) a one-line change.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings: defaults here, overridable via env vars / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets ---
    anthropic_api_key: str = ""  # from ANTHROPIC_API_KEY in .env

    # --- Chunking ---
    chunk_size: int = 1500               # target chunk length, in characters
    chunk_overlap: int = 300             # overlap between consecutive chunks
    chunk_break_threshold: float = 0.5   # only break at a boundary past this fraction of chunk_size

    # --- Retrieval ---
    top_k: int = 5                       # chunks retrieved for context
    rerank_candidate_multiplier: int = 2 # retrieve top_k * this many before reranking
    use_reranking_default: bool = False  # default state of the UI reranking toggle

    # --- Models ---
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    claude_model: str = "claude-sonnet-4-5-20250929"

    # --- LLM call ---
    max_tokens: int = 1024               # max tokens in Claude's answer
    temperature: float = 0.0             # 0.0 = deterministic; best for grounded Q&A
    max_question_length: int = 1000      # reject questions longer than this

    # --- Conversation memory ---
    history_exchanges: int = 3           # prior Q&A pairs sent to Claude for follow-ups

    # --- Vector store ---
    db_path: str = "data/chroma_db"
    collection_name: str = "documents"
    distance_metric: str = "cosine"

    # --- UI ---
    source_preview_chars: int = 300      # characters of each source chunk shown in the UI


# Single shared instance — import this everywhere: `from src.config import settings`
settings = Settings()