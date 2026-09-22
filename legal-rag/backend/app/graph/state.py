"""The shared state object threaded through every graph node.

WHY a single typed state dict: LangGraph nodes are pure functions of the form
``state -> partial_state_update``. Modeling the state explicitly (rather than
passing many positional args) means the LangGraph build and the framework-free
sequential fallback can drive the *exact same* node functions — proving the
pipeline really is a state machine, not framework glue. Each node reads the
keys it needs and writes the keys it produces; nothing else changes.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..schemas import Chunk, Citation, Conflict, SourceType


class GraphState(TypedDict, total=False):
    # --- input ---
    query: str                       # raw user query
    sanitized_query: str             # normalized, injection-stripped query
    top_k: int                       # effective rerank top-k for this request

    # --- routing ---
    routed_to: list[SourceType]      # corpora selected by the router

    # --- retrieval ---
    retrieved: dict[str, list[Chunk]]  # per-corpus fused candidates
    candidates: list[Chunk]            # all candidates merged across corpora
    reranked: list[Chunk]              # top-k after cross-encoder

    # --- merge + conflict ---
    merged: list[Chunk]              # deduped, authority-ordered sources
    conflicts: list[Conflict]        # detected disagreements

    # --- context + generation ---
    context_text: str                # numbered [S1]..[Sn] context block
    context_map: dict[str, Chunk]    # "S1" -> Chunk, for citation resolution
    answer: str                      # LLM answer text
    citations: list[Citation]        # parsed + validated citations

    # --- validation / control ---
    valid: bool                      # did the answer pass all checks?
    abstained: bool                  # did the model decline to answer?
    regenerate_count: int            # how many regeneration passes so far
    strict: bool                     # regenerate with a stricter prompt?
    needs_human_review: bool         # escalate to human review queue?
    warnings: list[str]              # non-fatal notes (e.g. injection flagged)
    errors: list[str]

    # --- carriers for pure-function nodes (injected once at graph build) ---
    _deps: dict[str, Any]            # store, llm, reranker, settings
