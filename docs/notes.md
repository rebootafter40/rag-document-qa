# RAG Document Q&A — Project Notes

Running notes for the project. Kept current throughout development so that
Week 10 documentation is easy to write.

---

## Week 1 — Environment Setup & Git Fundamentals

- Set up Git, GitHub repo, virtual environment, initial dependencies
- Created project structure with `src/`, `tests/`, `data/`, `notebooks/`
- Learned core Git workflow: init, add, commit, push

## Week 2 — PDF Ingestion & Chunking Pipeline

- Built `document_loader.py`: PDF text extraction with PyMuPDF (fitz)
- Built `chunker.py`: configurable text chunking with overlap
- Chunks carry metadata (source file, page number, chunk index)
- Smart break points: prefers splitting at newlines or sentence endings
  over mid-word splits

## Week 3 — Embeddings & Vector Store

- Built `embeddings.py`: sentence-transformers (all-MiniLM-L6-v2), runs locally
- Built `vector_store.py`: ChromaDB with cosine similarity
- Built `retriever.py`: clean interface combining embed + search
- End-to-end pipeline working: load → chunk → embed → store → retrieve

## Week 4 — LLM Integration & Answer Generation

- Built `qa_chain.py`: Claude Sonnet integration with grounding prompt
- System prompt enforces: answer only from context, cite sources, admit gaps
- Tested with multiple question types against the AI Action Plan PDF
- The system correctly refuses to answer questions outside document scope

## Week 5 — Streamlit Frontend

- Built `app.py`: file upload, chat interface, expandable source citations
- Session state management to avoid reprocessing
- Clear & upload new document functionality

## Week 6 — Improvements & Error Handling

### Bug Fix: Temp Filename in Citations
- **Problem**: Source citations showed `tmp3f8a2b.pdf` instead of original filename
- **Root cause**: `app.py` passed temp file path to `ingest_pdf()`, which flowed
  through to `load_pdf()` where `path.name` was used as the source metadata
- **Fix**: Added `original_filename` parameter to `load_pdf()` and `ingest_pdf()`,
  threaded from `app.py` where `uploaded_file.name` is available
- **Design pattern**: Backwards-compatible API extension — the parameter is
  optional with a sensible default, so CLI test scripts still work unchanged

### Error Handling
Added comprehensive error handling across the full pipeline:

- **document_loader.py**: File size limit (50 MB), page limit (200 pages),
  corrupt/password-protected PDF detection, image-based PDF detection (no
  extractable text)
- **vector_store.py**: Empty chunks validation, empty collection guard in
  query (returns [] instead of crashing), safe `min(top_k, doc_count)`
- **qa_chain.py**: Empty question validation, early API key check, empty
  retrieval results handling, specific catches for all Anthropic API errors
  (AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError,
  APIStatusError)
- **app.py**: try/except around ingestion with `st.error()` for user-friendly
  messages, `finally` block for temp file cleanup, try/except around `ask()`
  so errors appear as chat messages instead of stack traces, conditional
  sources expander (hidden when no sources)

Tested scenarios: corrupt PDF, missing API key, normal happy path — all passed.

### Structured Logging
- Created `src/logging_config.py` with centralized format:
  `%(asctime)s | %(name)s | %(levelname)s | %(message)s`
- Replaced all `print()` calls in pipeline modules with `logger.info()`
- Each module uses `logging.getLogger(__name__)` for clear source identification
- `__main__` test blocks intentionally kept `print()` for interactive output
- `app.py` calls `setup_logging()` once on startup

### Retrieval Quality Evaluation
Created `scripts/evaluate_retrieval.py` with:
- 10-question test set with expected pages and key terms
- 3 chunk configurations × 2 reranking modes = 6 total configurations
- Two metrics: page hit rate (did we retrieve from the right pages?) and
  term coverage (did the retrieved chunks contain the expected key terms?)

**Results:**

| Config              | Chunks | Page Hits | Term Coverage | Time  |
|---------------------|--------|-----------|---------------|-------|
| 500/100             | 205    | 100%      | 82%           | 3.4s  |
| 500/100 +rerank     | 205    | 100%      | 82%           | 6.2s  |
| 1000/200            | 105    | 90%       | 81%           | 1.5s  |
| 1000/200 +rerank    | 105    | 100%      | 88%           | 2.4s  |
| 1500/300            | 73     | 100%      | 98%           | 1.2s  |
| 1500/300 +rerank    | 73     | 100%      | 92%           | 2.6s  |

**Winner**: 1500/300 without reranking — best scores and fastest.

Key insight: the "three pillars" question that previously found only 2/3 pillars
now finds all 3 with the larger chunk size. The third pillar (International
Diplomacy and Security — PDF page 23; printed page 20, with the +3 offset coming
from the cover/epigraph/table-of-contents front matter) was getting missed with
1000-size chunks because it wasn't landing in the top 5 retrieved results.

### Reranking Module
- Built `src/reranker.py` using cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Cross-encoders process query+document together (more accurate than bi-encoders)
- Retrieve 2x candidates, rerank to top_k — improves accuracy for smaller chunks
- Added as optional toggle in the Streamlit sidebar under Settings
- Reranking helped the 1000-config most (+10% page hits, +7% term coverage)
- Reranking slightly hurt the 1500-config (-6% term coverage) — the bi-encoder
  was already good enough that the tighter focus excluded useful chunks

### Unit Tests
Created 20 pytest tests across 3 files, all passing:

- **test_chunker.py** (8 tests): basic chunking, metadata preservation,
  sequential indices, multi-page, short text → single chunk, empty input,
  overlap validation, no empty chunks
- **test_document_loader.py** (8 tests): 4 validation tests (file not found,
  wrong extension, corrupt PDF, filename override), 4 integration tests
  with real PDF (loads pages, has text, sequential page numbers, correct source)
- **test_qa_chain.py** (4 tests): build_context formatting — single result,
  multiple results, empty results, divider separation

## Red Team Audit (post-Week 6)

Ran an adversarial review of the project before starting Week 7.

**Fixed:**
- Updated model string from `claude-sonnet-4-20250514` to
  `claude-sonnet-4-5-20250929` — the old model was approaching retirement
  (May 2026); the new one is supported through at least September 2026.
  For a portfolio app that stays live for months, model longevity matters.
- Added a question length limit (1000 chars, now `max_question_length` in
  config) to prevent API cost abuse from oversized prompts

**Checked, confirmed OK:**
- requirements.txt current; ChromaDB on 1.5.0 (current 1.x line, pinned)

**Noted for awareness (no fix needed now):**
- Temp file orphans possible if Streamlit crashes mid-processing (minor)
- API client created at import time with a potentially null key — mitigated
  by the early key check in `ask()`
- evaluate_retrieval.py hardcodes the PDF path (fine for a one-time script)
- **Deployment concern for Week 9:** sentence-transformers model download
  (~80MB) plus local inference could hit memory limits on Streamlit Cloud's
  free tier

## Week 7 — Multi-Document Support & Conversation Memory

### Multi-Document Support
- Changed chunk IDs from `chunk_0` to `{source}_chunk_0` to prevent
  collisions across documents
- Added `delete_document()` to vector_store.py — removes all chunks
  for a specific file using ChromaDB's `where` filter on source metadata
- Updated app.py from single-file tracking (`processed_file`) to a list
  of dicts (`processed_files`) with name and chunk count per document
- Duplicate upload detection prevents processing the same file twice
- Sidebar shows all uploaded documents with individual remove buttons
- "Clear All Documents" replaces the old "Clear & Upload New"

### Conversation Memory
- Added `conversation_history` parameter to `ask()` in qa_chain.py
- Passes last 3 Q&A exchanges (6 messages) to Claude for follow-up context
- app.py sends `st.session_state["messages"][:-1]` to avoid duplicate
  user messages (current question is already appended before the call)
- Follow-ups like "which agencies handle those actions?" now work because
  Claude has prior context

### Key Learning
- Conversation memory helps the LLM understand follow-ups, but the
  retriever still operates on the raw query. Vague follow-ups like
  "tell me more about that" retrieve poorly because they lack semantic
  content for vector search. Query rewriting would fix this but is a
  stretch goal.
- RAG retrieval is semantic, not structural — asking for "page 3" doesn't
  work because vector search matches on meaning, not page numbers.

## Week 8 — Configuration & Evaluation

### Configuration Management (`src/config.py`)
- Centralized every tunable value plus the API key into a single validated
  `Settings` object using pydantic-settings
- Chose pydantic-settings over a plain dataclass/constants module: env-var
  and `.env` overrides are built in, which means zero rework when deploying
  in Week 9 (Streamlit Cloud secrets become one-line overrides)
- All six src modules now read from `from src.config import settings`
- Settings groups: secrets, chunking, retrieval, models, LLM call,
  conversation memory, vector store, UI — every option documented inline
- Modules keep optional parameters that default to `None` and resolve to
  config values inside the function — callers can still override per-call
  (backwards-compatible pattern from Week 6, applied project-wide)
- Set `temperature=0.0` explicitly (was silently riding the API default) —
  deterministic output is correct for grounded Q&A
- Extracted the chunker's hardcoded 0.5 break-point ratio into
  `chunk_break_threshold`

### Bugs Caught During the Refactor
- **chunk_size drift**: chunker defaulted to 1000 while retriever passed
  1500 — modules disagreed about the default. Centralizing config surfaced
  and fixed it. This is the quiet payoff of single-source-of-truth settings.
- **Stale-rewrite regression**: a wholesale rewrite of vector_store.py from
  an outdated copy silently reverted two Week 7 changes (source-scoped chunk
  IDs and `delete_document`). Both surfaced at app launch and were restored.
  Lesson: make targeted edits to the current file; never rewrite a module
  from a snapshot without diffing against the working version first.

### End-to-End QA Evaluation (`scripts/evaluate_qa.py`)
Distinct from Week 6's retrieval eval — this one scores the *answers*,
not just the retrieved chunks:

- 12-question test set against the America's AI Action Plan PDF
- Automated metrics: citation accuracy (cited page within ±1 of expected),
  keyword coverage (expected terms present in the answer), refusal check
  (out-of-scope questions correctly declined)
- Manual metric: answer correctness, scored by reading each answer

**Results:** citation accuracy 9/10, keyword coverage 17/20, refusals 2/2,
manual correctness 12/12. One documented miss (Q7, a structural query —
consistent with the known "semantic, not structural" retrieval limitation).
Full results in `docs/eval_results.md`.

### Known Limitation Documented
- `top_k` is a global budget across all loaded documents. With one document,
  the top 5 chunks all come from it; with five documents, those same 5 slots
  are shared, so per-document coverage degrades as the library grows.
  Per-document `top_k` budgeting deferred as a stretch goal.

### Code Cleanup
- Ran black (15 files reformatted) — purely cosmetic; all 20 tests passed after
- Ran ruff: removed 2 unused imports (leftover `retrieve` in app.py, unused
  `pytest` in a test) and 1 stray f-string with no placeholders
- Pinned black/ruff config in `pyproject.toml` (line length, target py312)
- All 20 tests green post-cleanup — confirms the formatting/lint changes were
  behavior-neutral

### Key Learnings
- Configuration centralization pays forward: the pydantic-settings choice
  means Week 9 deployment config is mostly "set environment variables"
- Refactoring is a bug-finding activity, not just cleanup — the chunk_size
  drift only became visible when everything read from one place
- Two kinds of evaluation, two kinds of confidence: retrieval eval answers
  "are the right chunks coming back?"; end-to-end QA eval answers "is the
  user getting a correct, cited answer?" A system can pass one and fail
  the other.
- Project knowledge hygiene matters as much as code hygiene: stale file
  copies (in a repo, an editor buffer, or an AI assistant's context) are
  how regressions happen

## Week 9 — Deployment to Streamlit Community Cloud

### Pre-Deployment Audit
- **Dependencies**: replaced the `pip freeze`-generated `requirements.txt`
  (~130 packages) with 6 direct dependencies. Removed dev tools
  (black/ruff/pytest) and the unused FastAPI stack (fastapi/uvicorn/starlette);
  left transitive deps (torch, numpy, tokenizers, etc.) for pip to resolve.
  Validated by installing into a fresh throwaway venv — all imports resolved.
- **Secrets**: confirmed `.env` was never committed to any branch
  (`git log --all -- .env` returned nothing) and is properly gitignored.
  The API key moves to Streamlit Cloud's secrets manager instead.
- **Memory**: the free tier caps at ~1 GB, and torch is the heavy hitter.
  Two levers applied: reranking stays off by default (the cross-encoder never
  loads on a fresh session), and CPU-only torch is pinned via
  `--extra-index-url https://download.pytorch.org/whl/cpu` (no GPU on the
  cloud, so the CUDA build would waste hundreds of MB). Fallback plan if it
  didn't fit: Hugging Face Spaces (more RAM, built for ML apps) or swapping to
  API-based embeddings so nothing heavy loads locally.

### Deployment
- Deployed to Streamlit Community Cloud from GitHub (main branch, `app.py`).
- API key set in Streamlit secrets as TOML: `ANTHROPIC_API_KEY = "..."`.
  pydantic-settings reads it from the environment with **zero code change** —
  the Week 8 config decision paid off exactly as intended.
- Build log confirmed the CPU-torch pin took effect (torch pulled from
  `download.pytorch.org/whl/cpu`) — the memory optimization we couldn't verify
  locally, verified on the actual Linux build.
- App booted cleanly and the free-tier memory **held** through the first
  question's model load (~80 MB embedding model download). The memory concern
  flagged in the Red Team audit did not materialize on this deploy.
- Verified end-to-end on the live URL: uploaded the AI Action Plan, asked
  "What are the three pillars?", got a correct, cited answer.
- Live URL: _(add your Streamlit app URL here)_

### Observation
- The three-pillars answer cited all three pillars as Page 3 — the Table of
  Contents, where the pillar titles are listed together and therefore the
  densest single source for that question. A correct citation, though it's the
  TOC rather than where each pillar's content begins (pages 6, 17, 23). A nice
  illustration of retrieval finding the highest-density match.

### Key Learnings
- `pip freeze` is not a deployment requirements file. It captures the whole
  environment; a deploy file should list only direct dependencies and let pip
  resolve the rest. Slimming it also reduces build time and memory pressure.
- Know the platform: CPU-only torch is the single biggest memory lever for an
  ML app on a GPU-less host. Shipping the CUDA build wastes hundreds of MB.
- The config groundwork pays off at deploy: because pydantic-settings reads
  environment variables, moving from a local `.env` to cloud secrets required
  no code change at all.
- Git hygiene (learned the hard way this session): edit files locally and let
  `git push` be the *only* path to GitHub — editing in the GitHub web UI
  creates a divergent commit and a merge conflict. And after saving a file
  handed to you, verify it's actually on disk (`grep` a known phrase) before
  committing, rather than assuming the save happened.

---

## Current Project Structure

```
rag-document-qa/
├── app.py                    # Streamlit frontend
├── README.md
├── requirements.txt          # 6 direct deps + CPU-torch index (Week 9)
├── pyproject.toml            # black + ruff config (Week 8)
├── .env.example
├── .gitignore
├── docs/
│   ├── notes.md              # This file
│   └── eval_results.md       # End-to-end QA eval results (Week 8)
├── src/
│   ├── __init__.py
│   ├── config.py             # Central pydantic-settings config (Week 8)
│   ├── logging_config.py     # Centralized logging setup (Week 6)
│   ├── document_loader.py    # PDF text extraction with PyMuPDF
│   ├── chunker.py            # Text chunking with configurable size/overlap
│   ├── embeddings.py         # Vector embeddings with sentence-transformers
│   ├── vector_store.py       # ChromaDB storage and search
│   ├── retriever.py          # Clean interface combining embed + search
│   ├── reranker.py           # Cross-encoder reranking (Week 6)
│   └── qa_chain.py           # Claude integration for answer generation
├── scripts/
│   ├── evaluate_retrieval.py # Retrieval quality evaluation (Week 6)
│   └── evaluate_qa.py        # End-to-end QA evaluation (Week 8)
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py       # 8 tests (Week 6)
│   ├── test_document_loader.py # 8 tests (Week 6)
│   └── test_qa_chain.py      # 4 tests (Week 6)
├── data/
│   ├── sample_docs/          # Test PDFs
│   └── chroma_db/            # Vector database storage (gitignored)
└── notebooks/
```

---

## Known Issues / Tech Debt

- **Streamlit Cloud memory (Week 9)**: sentence-transformers + torch load close
  to the free tier's ~1 GB ceiling. It held on the first deploy (verified live),
  but remains a watch item under heavier load or concurrent users. Mitigations
  in place: reranking off by default, CPU-only torch. Fallback if it degrades:
  Hugging Face Spaces or API-based embeddings.
- **Global `top_k` budget**: retrieval doesn't allocate slots per document;
  coverage of any single document degrades as more documents are loaded.
  Stretch goal: per-document budgeting.
- **Retrieval picks up page furniture**: near-empty pages (blank pages, header
  lines) and the Table of Contents sometimes surface as sources. Candidate fix:
  drop chunks below a minimum character length at ingest time.
- **Noisy third-party logging**: httpx and sentence-transformers emit many
  INFO-level lines during model loading. Handled locally in `evaluate_qa.py`
  (those loggers set to WARNING). Still to do: apply the same centrally in
  `logging_config.py` so it covers the app too.
- **Eval script re-ingestion**: evaluate_retrieval.py re-ingests the full
  document for each configuration (6×). Could re-embed only when chunk size
  changes, but not worth optimizing a one-time script.
- **Streamlit file watcher reload loop on Windows**: workaround is
  `--server.fileWatcherType none` or the `.streamlit/config.toml` fix.