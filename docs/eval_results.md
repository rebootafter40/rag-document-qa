# End-to-End QA Evaluation Results

_Document:_ `data/sample_docs/Americas-AI-Action-Plan.pdf`  
_Run:_ 2026-06-10 09:53  
_Page tolerance:_ ±1

| # | Question | Type | Cited | Expected | Citation | Keywords | Refusal | Correct |
|---|----------|------|-------|----------|----------|----------|---------|---------|
| 1 | What are the three pillars of America's AI Action Plan? | list | 4, 5, 27, 28 | 4, 5 | ✓ | 2/2 | — | ✓ |
| 2 | What does the plan say about open-source and open-weight ... | factual | 7, 8, 28 | 7, 8 | ✓ | 2/2 | — | ✓ |
| 3 | How does the plan propose to train the AI workforce? | synthesis | 9, 10, 20, 28 | 9, 10, 20 | ✓ | 2/2 | — | ✓ |
| 4 | What does the plan say about AI infrastructure and energy? | factual | 4, 17, 18, 19 | 17, 18, 19 | ✓ | 2/2 | — | ✓ |
| 5 | How does the plan address competition with China? | synthesis | 1, 3, 17, 23 | 4, 17, 23 | ✓ | 1/2 | — | ✓ |
| 6 | What does the plan say about AI and national security? | factual | 4, 5, 9, 28 | 9, 13, 24 | ✓ | 1/1 | — | ✓ |
| 7 | What is Pillar III about? | factual | 3, 4, 12, 13 | 23 | ✗ | 2/2 | — | ✓ |
| 8 | What does the plan recommend about regulatory barriers to... | factual | 4, 6, 7, 8 | 6, 7 | ✓ | 1/1 | — | ✓ |
| 9 | What role does the federal government play in supporting ... | factual | 6, 7, 8, 24 | 7, 8 | ✓ | 2/2 | — | ✓ |
| 10 | How does the plan support displaced or retrained workers? | factual | 9, 10, 20 | 10 | ✓ | 2/2 | — | ✓ |
| 11 | What is the capital of France? | negative | 1, 3, 4, 24 | refuse | — | — | ✓ | ✓ |
| 12 | What's a good recipe for chocolate chip cookies? | negative | 5, 19, 24, 27 | refuse | — | — | ✓ | ✓ |

## Summary

- **Citation accuracy:** 9/10 (90%)
- **Keyword coverage:** 17/18 (94%)
- **Refusal accuracy:** 2/2 (100%)
- **Manual correctness:** 12/12 (100%)

## Notes & Findings

- **Q7 ("What is Pillar III about?") — citation miss, graceful degradation.**
  Expected page 23 (where Pillar III begins) did not reach the top-k for this
  query, so the system answered only from what it retrieved — and correctly
  *flagged the gap* rather than fabricating the section's contents. This is a
  retrieval miss, not a hallucination: retrieval quality and answer quality are
  separate, and the end-to-end eval surfaced both. Likely cause: "Pillar III"
  is a structural reference, and vector search matches meaning, not document
  structure (same root cause as page-number lookups failing).

- **Negative tests (Q11, Q12) — anti-hallucination held under pressure.** Q11
  ("capital of France") is adversarial: the document mentions the "Paris AI
  Action Summit." The system noticed "Paris" was present, reasoned it appeared
  only as a summit location, and declined to answer anyway.

- **Retrieval noise:** near-empty pages 27–28 (blank page / header) repeatedly
  appear as retrieved sources, consuming top-k slots. Candidate fix (future):
  drop chunks below a minimum character length at ingest time.

- **Keyword coverage is a vocabulary proxy, not a correctness measure** — which
  is why correctness is scored manually. (Q5 scored 1/2 keywords but the answer
  was correct.)