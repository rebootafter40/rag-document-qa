# Project Notes

These are working notes collected during development. They'll be
organized into proper documentation later.

## Decisions Made
- Chose ChromaDB over Pinecone: simpler setup, no cloud dependency, good for learning
- Chose PyMuPDF over pdfplumber: faster, handles more PDF types
- Building from scratch (not LangChain): shows deeper understanding for portfolio
- Chose chunk_size=1000 with overlap=200 as defaults: good balance between
  precision and context (105 chunks from 28-page doc)
- Chose sentence-transformers (all-MiniLM-L6-v2) over OpenAI embeddings:
  free, runs locally, good quality results
- Using Claude Sonnet for QA with strict grounding prompt: keeps costs low,
  prevents hallucination

## Things I Learned
- Smaller chunks give more precise retrieval but less context. Larger chunks
  include more context but may dilute relevance. Overlap ensures sentences
  at boundaries aren't lost.
- Cosine distance below 0.35 generally indicated strong matches in testing.
  System correctly distinguished related vs unrelated content.
- The system prompt is critical for RAG quality. Instructing Claude to "only
  answer based on provided context" and "cite source and page number"
  dramatically reduces hallucination.
- The app correctly refuses to answer questions outside the document's scope.
  When asked about France's AI policy using a US policy document, it admitted
  it couldn't answer, noted the only France-related mention it found, and
  explained what would be needed. This is a direct result of the system
  prompt design.
  Good grounding prompts make the system admit gaps rather than hallucinate. When retrieval misses relevant chunks, the model says 'I don't know' instead of guessing — this is a feature, not bug.

## Issues Encountered
- ChromaDB doesn't support Python 3.14 yet, had to install Python 3.12
- Temp file naming causes source citations to show "tmpXXX.pdf" instead
  of original filename (cosmetic, needs fix)

## Future Improvements
- Fix temp filename in source citations
- Test with different document types
- Add input validation and error handling
