# Legal RAG

A production-shaped **retrieval-augmented generation** pipeline that answers US
legal questions from three corpora — **Federal statutes (U.S. Code)**, the
**CFR**, and **case law** — and **grounds every factual sentence in a
verifiable citation**, regenerating under a stricter prompt (or escalating to a
human review queue) when citation validation fails.

The whole system is a **LangGraph state machine** (with a framework-free
sequential fallback that runs the identical nodes), and it **runs fully offline
with zero downloads**: when no API key or model is available it degrades to a
deterministic mock LLM and mock embedder, so `demo.py`, the tests, and the eval
gate all produce real, reproducible output on a bare Python 3.11.

```
Next.js → FastAPI → input validation → Router → per-corpus hybrid retrieval
(BM25 + Vector) → RRF fusion → cross-encoder rerank → source merge →
conflict builder → context builder → LLM → citation + grounding validation
→ (valid) answer  |  (invalid & under budget) regenerate  |  (else) human review
```

## Run offline in 3 commands

No API keys, no model downloads, no external services. From the repo root:

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python backend/demo.py          # end-to-end pipeline, prints every stage
cd backend && python -m pytest -q && cd .. && python eval/run_eval.py
```

`demo.py` prints, per query: which corpora it `routed_to`, candidate/top-k
counts, detected conflicts, the answer, the parsed citations with **verified**
flags, and whether the answer was valid / how many regenerations it took.
`run_eval.py` prints a **PASS/FAIL promotion gate** over a golden set.

Serve the API:

```bash
cd backend && uvicorn app.main:app --reload   # http://localhost:8000
curl localhost:8000/health
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query":"What is the deadline to file an EEOC charge?"}'
```

Or with Docker: `docker compose up api` (add `--profile qdrant` for the vector DB).

## What makes it trustworthy

- **Hybrid retrieval + RRF.** BM25 catches exact legal tokens ("42 U.S.C.
  1983", "180 days"); dense vectors catch paraphrase/intent. Reciprocal Rank
  Fusion (k=60) combines them scale-free, with no weight tuning.
- **Cross-encoder rerank.** A precision stage over the fused candidates keeps
  only the top-K passages the answer is allowed to use.
- **Conflict detection.** The pipeline *surfaces* disagreements instead of
  averaging them: differing filing deadlines (the seed data has a real
  **180-day statute vs 300-day CFR** conflict), mandatory-vs-permissive
  language, and superseded/overruled markers (the seed data has **Monroe v.
  Pape**, overruled by **Monell**).
- **Grounded generation.** The model may answer *only* from numbered `[S#]`
  passages, must cite every factual sentence, and must abstain when unsupported.
- **Validation as a gate.** Every `[S#]` must resolve to a retrieved passage,
  each cited sentence must be supported by its passage, and no factual sentence
  may be uncited. Failure → regenerate (stricter) up to `MAX_REGENERATE`, else
  `valid=false` and the response is flagged for human review.

## Mock → production swap table

Everything below runs offline via a deterministic mock and swaps to the real
component behind one flag or dependency — the interface never changes.

| Concern | Offline default (this repo) | Production swap | How to switch |
|---|---|---|---|
| **Embeddings** | `MockEmbedder` (pure-python hashing, 256-d) | `sentence-transformers` all-MiniLM-L6-v2 | `pip install sentence-transformers` + `USE_REAL_EMBEDDINGS=1` |
| **Reranker** | `MockCrossEncoder` (token-overlap) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `pip install sentence-transformers` + `USE_REAL_RERANKER=1` |
| **LLM** | `MockLLM` (deterministic, context-grounded) | Anthropic `claude-3-5-sonnet` (default) / OpenAI | `pip install anthropic` + `USE_REAL_LLM=1` + `ANTHROPIC_API_KEY` |
| **Vector store** | `NumpyVectorIndex` (in-memory) | `QdrantVectorIndex` | `pip install qdrant-client` + `VECTOR_BACKEND=qdrant` (+ `docker compose --profile qdrant up`) |
| **BM25** | pure-python Okapi BM25 fallback | `rank-bm25` | `pip install rank-bm25` (already in requirements) |
| **Graph engine** | `SequentialRunner` (framework-free) | `langgraph.StateGraph` | `pip install langgraph` (already in requirements) |
| **Grounding check** | content-term coverage heuristic | NLI / LLM entailment | swap the check in `nodes/validate_answer.py` |
| **Corpora** | small public-domain seed JSON in `data/` | U.S. Code / CFR / CAP bulk data | see `backend/app/ingest/fetch_public_data.py` |

All tunables are env-driven (`BM25_TOP_N`, `VECTOR_TOP_N`, `RRF_K`,
`RERANK_TOP_K`, `MAX_REGENERATE`, model names, `VECTOR_BACKEND`). See
[`.env.example`](./.env.example).

## Layout

```
legal-rag/
  backend/
    demo.py                      # offline end-to-end driver (prints each stage)
    app/
      main.py                    # FastAPI: POST /query, GET /health
      schemas.py  config.py      # Pydantic v2 contracts + env-driven config
      text_utils.py              # legal-aware sentence splitting / term overlap
      graph/
        state.py build_graph.py  # StateGraph + regenerate edge + fallback runner
        nodes/                   # one file per pipeline box (see ARCHITECTURE.md)
      retrieval/                 # indexes.py (BM25 + numpy/qdrant), rrf.py, reranker.py
      llm/                       # provider-agnostic wrapper + mock
      ingest/                    # build_indexes.py, fetch_public_data.py (documented)
    tests/                       # pytest: rrf, router, citation validation, e2e
  data/{federal,cfr,caselaw}/seed.json   # public-domain seed passages + a real conflict
  eval/                          # golden.jsonl + run_eval.py (PASS/FAIL gate)
  frontend/                      # Next.js single-page stub (answer + clickable [S#])
  docker-compose.yml  Dockerfile  requirements.txt  .env.example  ARCHITECTURE.md
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for a node-by-node walkthrough and how
each feature maps to a legal-AI platform's needs.

## Data & licensing

The seed passages under `data/` are US government works (statutes, regulations,
judicial opinions), which are not subject to federal copyright. Nothing is
scraped at build time. To ingest real corpora at scale, see the documented
loaders and official bulk-data URLs in
[`backend/app/ingest/fetch_public_data.py`](./backend/app/ingest/fetch_public_data.py)
(OLRC U.S. Code XML, GovInfo CFR bulk data, Caselaw Access Project /
CourtListener).

> This is an engineering demo, not legal advice.
