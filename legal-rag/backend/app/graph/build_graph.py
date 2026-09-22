"""Assemble the pipeline as a state machine.

Two interchangeable drivers run the SAME node functions:

  * :func:`build_langgraph` — a real ``langgraph.StateGraph`` with a conditional
    edge from ``validate_answer`` back to ``generate`` (the regenerate loop) or
    to END.
  * :class:`SequentialRunner` — a dependency-free driver that executes the nodes
    in order and implements the same conditional loop in plain Python.

WHY both: LangGraph gives us durable, inspectable graph execution in production
(checkpoints, streaming, retries). But the pipeline's correctness must not
depend on the framework being installed — so the fallback proves the design is a
genuine state machine and keeps the system runnable with zero third-party
packages. ``run_pipeline`` picks LangGraph when importable, else the fallback;
both produce an identical :class:`QueryResponse`.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..config import Settings
from ..llm import get_llm, llm_mode
from ..retrieval.indexes import CorpusStore
from ..retrieval.reranker import get_reranker
from ..schemas import QueryResponse, SourceType
from . import nodes
from .state import GraphState

# Ordered node pipeline shared by both drivers. Each entry: (name, fn).
_PIPELINE: list[tuple[str, Callable[[GraphState], GraphState]]] = [
    ("validate_input", nodes.validate_input),
    ("route", nodes.route),
    ("retrieve", nodes.retrieve),
    ("rerank", nodes.rerank),
    ("merge_conflict", nodes.merge_conflict),
    ("build_context", nodes.build_context),
    ("generate", nodes.generate),
    ("validate_answer", nodes.validate_answer),
]


def _make_deps(store: CorpusStore, settings: Settings) -> dict:
    return {
        "store": store,
        "settings": settings,
        "llm": get_llm(settings),
        "reranker": get_reranker(settings),
    }


# --------------------------------------------------------------------------- #
# Framework-free sequential runner (the fallback / proof of state machine)
# --------------------------------------------------------------------------- #
class SequentialRunner:
    """Runs the nodes in order, looping generate<->validate on failure."""

    def __init__(self, store: CorpusStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    def invoke(self, state: GraphState) -> GraphState:
        state.setdefault("_deps", _make_deps(self.store, self.settings))
        state.setdefault("regenerate_count", 0)
        state.setdefault("warnings", [])
        state.setdefault("errors", [])

        # Run everything up to (and including) the first validate_answer.
        for name, fn in _PIPELINE:
            state.update(fn(state))
            if name == "validate_input" and state.get("errors"):
                # Unsalvageable input: short-circuit with an abstention.
                state.update(
                    answer="I cannot process this query.",
                    valid=False,
                    needs_human_review=False,
                    routed_to=[],
                    candidates=[],
                    reranked=[],
                    merged=[],
                    conflicts=[],
                    citations=[],
                    context_map={},
                )
                return state

        # Regenerate loop: re-run generate + validate under stricter prompt.
        while nodes.should_regenerate(state) == "regenerate":
            state.update(nodes.generate(state))
            state.update(nodes.validate_answer(state))

        return state


# --------------------------------------------------------------------------- #
# LangGraph driver
# --------------------------------------------------------------------------- #
def build_langgraph(store: CorpusStore, settings: Settings):
    """Build a compiled LangGraph StateGraph, or return None if unavailable."""
    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except Exception:
        return None

    deps = _make_deps(store, settings)

    def _wrap(fn: Callable[[GraphState], GraphState]):
        # Ensure deps ride along in state for every node.
        def _node(state: GraphState) -> GraphState:
            state.setdefault("_deps", deps)
            return fn(state)

        return _node

    def _abort(state: GraphState) -> GraphState:
        # Mirror the sequential runner's short-circuit for unusable input.
        return {
            "answer": "I cannot process this query.",
            "valid": False,
            "needs_human_review": False,
            "routed_to": [],
            "candidates": [],
            "reranked": [],
            "merged": [],
            "conflicts": [],
            "citations": [],
            "context_map": {},
        }

    def _after_input(state: GraphState) -> str:
        return "abort" if state.get("errors") else "route"

    graph = StateGraph(GraphState)
    for name, fn in _PIPELINE:
        graph.add_node(name, _wrap(fn))
    graph.add_node("abort", _abort)

    graph.set_entry_point("validate_input")
    graph.add_conditional_edges(
        "validate_input", _after_input, {"abort": "abort", "route": "route"}
    )
    graph.add_edge("abort", END)
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "merge_conflict")
    graph.add_edge("merge_conflict", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "validate_answer")
    graph.add_conditional_edges(
        "validate_answer",
        nodes.should_regenerate,
        {"regenerate": "generate", "end": END},
    )
    return graph.compile()


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _state_to_response(query: str, state: GraphState, settings: Settings) -> QueryResponse:
    return QueryResponse(
        query=query,
        routed_to=state.get("routed_to", []),
        answer=state.get("answer", ""),
        citations=state.get("citations", []),
        conflicts=state.get("conflicts", []),
        valid=bool(state.get("valid", False)),
        abstained=bool(state.get("abstained", False)),
        regenerated=int(state.get("regenerate_count", 0)),
        candidate_count=len(state.get("candidates", [])),
        top_k_count=len(state.get("merged", [])),
        needs_human_review=bool(state.get("needs_human_review", False)),
        warnings=state.get("warnings", []),
    )


def run_pipeline(
    query: str,
    store: CorpusStore,
    settings: Settings,
    top_k: Optional[int] = None,
    prefer_langgraph: bool = True,
) -> QueryResponse:
    """Run one query through the graph, returning a validated response.

    Picks the LangGraph driver when it is importable; otherwise uses the
    framework-free sequential runner. Both execute the identical nodes.
    """

    init: GraphState = {"query": query}
    if top_k is not None:
        init["top_k"] = top_k

    compiled = build_langgraph(store, settings) if prefer_langgraph else None
    if compiled is not None:
        final = compiled.invoke(init)
    else:
        final = SequentialRunner(store, settings).invoke(init)

    return _state_to_response(query, final, settings)


def engine_name(prefer_langgraph: bool = True) -> str:
    """Report which driver run_pipeline would use (for demos/health)."""
    if prefer_langgraph:
        try:
            import langgraph  # type: ignore  # noqa: F401

            return "langgraph"
        except Exception:
            return "sequential-fallback"
    return "sequential-fallback"
