# Legal RAG — Frontend (Next.js stub)

A minimal single-page Next.js client: it POSTs a question to the FastAPI
backend's `POST /query` and renders the grounded answer with **clickable `[S#]`
citations** that scroll to / highlight the corresponding source, plus any
detected conflicts and a validity badge.

This is intentionally a stub — one page, no styling framework — so it documents
the API contract the real UI would build on. The production UI would add
authentication, a source viewer with page/paragraph (and eventually
bounding-box) deep-linking, streaming answers, and a review-queue view for
answers where `valid=false`.

## Run

```bash
# 1. Start the backend (see the repo root README) on http://localhost:8000
# 2. Then:
npx create-next-app@latest legal-rag-ui --ts --app --no-tailwind
# copy app/page.tsx below into legal-rag-ui/app/page.tsx
cd legal-rag-ui
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## The whole page (`app/page.tsx`)

The reference implementation lives at [`app/page.tsx`](./app/page.tsx) in this
folder. It:

- keeps the question in local state and POSTs it to `${NEXT_PUBLIC_API_URL}/query`;
- renders `answer`, splitting on `[S#]` markers and turning each into a
  clickable anchor;
- lists `citations` with their `verified` flag and `page`/`paragraph`
  provenance, and `conflicts`;
- shows a red banner when `valid === false` (routed to human review).

## API contract (from `backend/app/schemas.py`)

Request:

```json
{ "query": "What is the EEOC charge filing deadline?", "top_k": 6 }
```

Response (abridged):

```json
{
  "query": "...",
  "routed_to": ["federal", "cfr"],
  "answer": "... within 180 days ... [S1]. ... within 300 days ... [S4].",
  "citations": [
    { "marker": "S1", "source_id": "42-usc-2000e-5-e1", "title": "42 U.S.C. § 2000e-5(e)(1)",
      "page": 12, "paragraph": "e(1)", "verified": true }
  ],
  "conflicts": [
    { "kind": "deadline", "description": "Sources state different filing deadlines: [S1] = 180 days, [S4] = 300 days ...", "markers": ["[S1]", "[S4]"] }
  ],
  "valid": true,
  "abstained": false,
  "regenerated": 0,
  "needs_human_review": false
}
```
