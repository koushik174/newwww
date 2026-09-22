"""End-to-end offline demo: prints every stage of the pipeline.

Run from the backend/ directory (or anywhere with backend on the path):

    python backend/demo.py

It builds the indexes from the seed corpora, runs a handful of queries fully
offline (mock LLM + mock embedder, zero downloads), and prints for each: which
corpora it routed to, candidate/top-k counts, the detected conflicts, the
generated answer, the parsed citations with their verified flags, and whether
the answer was valid / how many regenerations it took.
"""

from __future__ import annotations

import os
import sys

# Make `app` importable whether run from repo root or backend/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings  # noqa: E402
from app.graph.build_graph import engine_name, run_pipeline  # noqa: E402
from app.ingest.build_indexes import build_store  # noqa: E402
from app.llm import llm_mode  # noqa: E402

DEMO_QUERIES = [
    "What is the deadline to file an EEOC charge for employment discrimination?",
    "Can a city be sued under 42 U.S.C. 1983 and does it have qualified immunity?",
    "Is a municipality a person that can be sued for civil rights violations?",
]


def _hr(char: str = "=") -> None:
    print(char * 78)


def run() -> None:
    settings = get_settings()
    store = build_store(settings)

    _hr()
    print("LEGAL RAG — OFFLINE END-TO-END DEMO")
    print(f"engine        : {engine_name()}")
    print(f"llm_mode      : {llm_mode(settings)}")
    print(f"embedding_mode: {'real' if settings.use_real_embeddings else 'mock'}")
    print(f"corpora       : {store.counts()}")
    print(
        f"config        : bm25_top_n={settings.bm25_top_n} "
        f"vector_top_n={settings.vector_top_n} rrf_k={settings.rrf_k} "
        f"rerank_top_k={settings.rerank_top_k} max_regenerate={settings.max_regenerate}"
    )
    _hr()

    for i, q in enumerate(DEMO_QUERIES, 1):
        resp = run_pipeline(q, store, settings)
        print(f"\nQUERY {i}: {q}")
        _hr("-")
        print(f"routed_to        : {[s.value for s in resp.routed_to]}")
        print(f"candidate_count  : {resp.candidate_count}")
        print(f"top_k_count      : {resp.top_k_count}")
        print(f"valid            : {resp.valid}")
        print(f"abstained        : {resp.abstained}")
        print(f"regenerated      : {resp.regenerated}")
        print(f"needs_human_review: {resp.needs_human_review}")

        if resp.conflicts:
            print("conflicts        :")
            for c in resp.conflicts:
                print(f"   - [{c.kind}] {c.description}")
        else:
            print("conflicts        : (none)")

        print("answer           :")
        for line in resp.answer.split("\n"):
            if line.strip():
                print(f"   {line.strip()}")

        print("citations        :")
        for cit in resp.citations:
            flag = "✓ verified" if cit.verified else "✗ unverified"
            loc = []
            if cit.page is not None:
                loc.append(f"p.{cit.page}")
            if cit.paragraph:
                loc.append(f"¶{cit.paragraph}")
            loc_str = f" ({', '.join(loc)})" if loc else ""
            print(f"   [{cit.marker}] {flag} — {cit.title}{loc_str}")

        if resp.warnings:
            print("warnings         :")
            for w in resp.warnings:
                print(f"   - {w}")

    _hr()
    print("DEMO COMPLETE — ran fully offline with zero downloads.")
    _hr()


if __name__ == "__main__":
    run()
