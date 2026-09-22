# Architecture

Legal RAG is implemented as a **state machine**: a fixed set of pure-function
nodes that each read some keys of a shared `GraphState` and write others. The
same nodes are driven either by a real `langgraph.StateGraph` or by a
framework-free `SequentialRunner` — proving the design *is* a state machine, not
framework glue. This document walks every node in order and explains the *why*
behind each design choice, then maps the features onto what a production
legal-AI platform needs.

```
                ┌─────────────────┐
   query  ─────▶│ validate_input  │  Pydantic + injection normalization
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │      route      │  classify → {federal, cfr, caselaw}
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │    retrieve     │  per corpus: BM25 ∥ Vector → RRF fuse
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │     rerank      │  cross-encoder → Top-K
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ merge_conflict  │  dedupe + authority order; detect conflicts
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │  build_context  │  numbered [S1..Sn] + provenance + conflicts
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │    generate     │◀──────────────┐  answer only from context
                └────────┬────────┘               │  (stricter on regenerate)
                         ▼                         │
                ┌─────────────────┐   invalid &    │
                │ validate_answer │──under budget──┘
                └────────┬────────┘
                 valid │  │ exhausted budget
                       ▼  ▼
                   return    needs_human_review = true
```

## Shared state

`app/graph/state.py` defines `GraphState` (a `TypedDict`). Nodes are
`GraphState -> partial update`. Dependencies (the corpus store, LLM, reranker,
settings) are injected once into `state["_deps"]` so nodes stay pure and both
drivers can run them unchanged.

## Node-by-node

### 1. `validate_input` — `nodes/validate_input.py`
Enforces the Pydantic contract, normalizes control characters/whitespace, and
neutralizes prompt-injection directives ("ignore previous instructions", "reveal
your system prompt"). **Why:** user input is both a correctness and a security
surface. We *strip* the injected directive and keep the substantive question
(recording a warning) rather than over-blocking real legal queries. A query that
is *only* an injection short-circuits to a safe refusal.

### 2. `route` — `nodes/route.py`
Classifies the query to one or more corpora with transparent keyword rules,
falling back to *all three* when signals are weak. **Why:** retrieving from every
corpus dilutes the candidate pool and can bury the right passage after fusion; a
statute question should not be answered mainly from case law. Routing is
deliberately recall-oriented (multi-corpus on ambiguity). Production swap: an LLM
or small classifier behind the same `route` signature.

### 3. `retrieve` — `nodes/retrieve.py` (+ `retrieval/indexes.py`, `retrieval/rrf.py`)
For each routed corpus, runs **BM25** (lexical) and the **Vector** index (dense)
independently and fuses their ranked lists with **Reciprocal Rank Fusion**.
**Why hybrid:** legal queries mix exact tokens that must match ("180 days", a
section number) with semantic intent; BM25 owns the former, dense vectors the
latter, and neither alone gives sufficient recall. **Why RRF over a weighted
sum:** BM25 scores and cosine similarities are on different, unnormalized scales,
so summing them needs brittle per-corpus weights. RRF fuses on *rank* alone
(`score = Σ 1/(k+rank)`, k=60) — scale-free and tuning-free. Both index types
expose the identical `.search(query, top_n) -> list[Chunk]` interface, so the
in-memory numpy store and the optional Qdrant backend are interchangeable.

### 4. `rerank` — `nodes/rerank.py` (+ `retrieval/reranker.py`)
A **cross-encoder** rescores each `(query, passage)` pair jointly and keeps the
top-K (`RERANK_TOP_K`, default 6). **Why a separate stage:** BM25 and dense
retrieval are bi-encoders — fast and scalable but imprecise, because query and
document are embedded independently. A cross-encoder reads the pair together and
is far more accurate, but too expensive to run over a whole corpus. The standard
pattern is cheap recall (retrieve+fuse) → expensive precision (rerank a few
dozen) → small, focused context. This is where most end-to-end quality comes
from, and trimming to a small set is what makes ungrounded claims easy to catch.

### 5. `merge_conflict` — `nodes/merge_conflict.py`
Two responsibilities. **Merge:** dedupe passages and order them by legal
authority — statute (U.S. Code) > regulation (CFR) > case law — because that is
the order a lawyer weighs them and it biases resolution toward the controlling
source. **Conflict builder:** *detect and surface* disagreement rather than hide
it, via three checkable signals: (1) differing deadlines/day-counts for the same
action (the seed data's 180-day statute vs 300-day CFR), (2) mandatory
("shall"/"must") vs permissive ("may") language on the same obligation, and (3)
superseded/overruled markers carried in a passage's metadata (Monroe v. Pape,
overruled by Monell). **Why a dedicated node:** a RAG system that silently
averages conflicting sources is dangerous; conflicts must be made explicit so the
model confronts them.

### 6. `build_context` — `nodes/context.py`
Builds the numbered `[S1]..[Sn]` context block, carrying each passage's title,
citation, page and paragraph, and keeps a `context_map` (`"S1" -> Chunk`) so the
answer's markers resolve straight back to source objects. Detected conflicts are
injected as `CONFLICT:` lines with an instruction to resolve them. **Why:**
grounding is only auditable if every passage the model sees has a stable,
resolvable handle; the provenance fields are the hook for coordinate/bounding-box
citation once documents are ingested from PDFs.

### 7. `generate` — `nodes/generate.py` (+ `llm/`)
Calls the provider-agnostic LLM with a strict system prompt: answer **only** from
the numbered sources, cite **every** factual sentence with `[S#]`, address every
listed conflict, and **abstain** when unsupported. On a regenerate pass the
prompt is escalated (`STRICT_MODE`), which is what makes the loop corrective
rather than repetitive. **Why the mock:** with no key/model the `MockLLM`
deterministically emits one grounded sentence per relevant passage (the passage's
own leading sentence, tagged with its marker) and a conflict-resolution sentence,
so the entire system runs offline and every test/demo is reproducible.

### 8. `validate_answer` — `nodes/validate_answer.py`
The trust gate. Three checks: **(a) resolution** — every `[S#]` resolves to a
retrieved passage (a marker to nowhere is a fabricated citation); **(b)
grounding** — each cited factual sentence must be supported by the union of its
cited passages (offline: content-term coverage ≥ 0.5; production swap: NLI/LLM
entailment); a summary/conflict sentence that only references already-grounded
markers is accepted because it makes no new claim; **(c) no uncited factual
sentence** — every substantive sentence carries a marker (abstentions exempt). If
anything fails and we are under `MAX_REGENERATE`, set `strict` and loop back to
`generate`; if the budget is exhausted, return `valid=false` and set
`needs_human_review`. **Why loop then escalate:** one stricter retry recovers
most fixable answers cheaply, and refusing to emit an ungrounded legal answer is
the correct failure mode.

### Regenerate edge
`should_regenerate` is the conditional edge (`validate_answer → generate` or
`→ END`). In LangGraph it is a real conditional edge; in the fallback it is a
`while` loop calling the same `generate`/`validate_answer` functions.

## Retrieval internals

- `retrieval/indexes.py` — `BM25Index` (uses `rank-bm25`, else a pure-python
  Okapi BM25), `NumpyVectorIndex` (numpy or pure-python cosine),
  `QdrantVectorIndex` (optional, same interface), and `CorpusStore` that bundles
  a BM25 + Vector index per corpus and is built once at startup.
- `retrieval/rrf.py` — rank-based fusion with dedup by chunk key.
- `retrieval/reranker.py` — `MockCrossEncoder` and the real sentence-transformers
  `CrossEncoder`, chosen by `USE_REAL_RERANKER`.
- `embeddings.py` — `MockEmbedder` (hashing) and `SentenceTransformerEmbedder`,
  chosen by `USE_REAL_EMBEDDINGS`, behind one `embed(texts)` interface.

## Evaluation as a CI gate

`eval/run_eval.py` runs a golden set (`eval/golden.jsonl`) through the full
pipeline and computes **retrieval_recall**, **citation_accuracy**, and
**grounded_rate**, then prints a **PASS/FAIL promotion gate** and exits non-zero
on FAIL — ready to wire into CI so a change that degrades grounding cannot ship.

## Mapping to a legal-AI platform's needs

| Platform need | Where it lives here | Production trajectory |
|---|---|---|
| **Background agents** | The graph is a resumable state machine with a regenerate loop | LangGraph checkpoints + a queue worker running graphs asynchronously; the `SequentialRunner` shows the logic is framework-independent |
| **Hybrid RAG at document scale** | BM25 + dense + RRF + cross-encoder, per-corpus | Swap numpy → Qdrant (`VECTOR_BACKEND=qdrant`), real embeddings/reranker; shard indexes per corpus/jurisdiction |
| **Page / paragraph / bbox citations** | `Chunk.page`/`paragraph`; provenance carried into context and citations | Ingest PDFs with layout, attach bounding boxes to `Chunk.metadata`; the citation contract already resolves markers → source objects |
| **Conflict / authority reasoning** | `merge_conflict` (deadline, mandatory/permissive, overruled) + authority ordering | Add a subsequent-history graph (KeyCite-style), effective-date logic, jurisdiction scoping |
| **Eval as a CI gate** | `eval/run_eval.py` PASS/FAIL promotion gate | Expand golden sets per practice area; gate deploys; track drift over time |
| **Review queue + escalation** | `validate_answer` sets `valid=false` + `needs_human_review` after exhausting regenerations | Persist flagged answers to a queue, human-in-the-loop UI, feed corrections back into eval |
| **MCP server (next step)** | The FastAPI `/query` + typed schemas are a clean tool boundary | Expose the pipeline as an MCP tool so other agents can call grounded legal retrieval as a capability |

## Why this shape, in one line each

- **Hybrid retrieval** — exact tokens *and* semantics; neither alone recalls enough.
- **RRF over weighted sum** — scale-free rank fusion, no per-corpus weight tuning.
- **Separate cross-encoder** — precision the bi-encoders can't give, only over a small candidate set.
- **Conflict node** — surface legal disagreement instead of averaging it away.
- **Regenerate loop** — cheaply recover fixable answers; escalate the rest to humans.
- **LangGraph + fallback** — prove it's a state machine, keep it runnable with zero deps.
- **Mocks everywhere** — reproducible, offline, downloadable-free CI and demos.
