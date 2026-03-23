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
Diplomacy and Security, page 20) was getting missed with 1000-size chunks
because it wasn't landing in the top 5 retrieved results.

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

---

## Current Project Structure

```
rag-document-qa/
├── app.py                    # Streamlit frontend
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
│   └── notes.md              # This file
├── src/
│   ├── __init__.py
│   ├── logging_config.py     # Centralized logging setup (Week 6)
│   ├── document_loader.py    # PDF text extraction with PyMuPDF
│   ├── chunker.py            # Text chunking with configurable size/overlap
│   ├── embeddings.py         # Vector embeddings with sentence-transformers
│   ├── vector_store.py       # ChromaDB storage and search
│   ├── retriever.py          # Clean interface combining embed + search
│   ├── reranker.py           # Cross-encoder reranking (Week 6)
│   └── qa_chain.py           # Claude integration for answer generation
├── scripts/
│   └── evaluate_retrieval.py # Retrieval quality evaluation (Week 6)
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