"""Tests for the answer-validation trust gate and the regenerate loop."""

from __future__ import annotations

from app.config import Settings
from app.graph.build_graph import SequentialRunner
from app.graph.nodes.validate_answer import validate_answer
from app.llm import MockLLM
from app.schemas import Chunk, SourceType


def _ctx():
    c1 = Chunk(
        source_id="42-usc-1983",
        source_type=SourceType.FEDERAL,
        title="42 U.S.C. 1983",
        text="Every person who under color of law deprives another of rights secured "
        "by the Constitution shall be liable to the party injured.",
        paragraph="a",
    )
    c2 = Chunk(
        source_id="owen-1980",
        source_type=SourceType.CASELAW,
        title="Owen v. City of Independence",
        text="A municipality may not assert qualified immunity as a defense to "
        "liability under section 1983.",
        paragraph="IV",
    )
    return {"S1": c1, "S2": c2}


def _state(answer: str, regen: int = 0):
    settings = Settings.from_env()
    return {
        "answer": answer,
        "context_map": _ctx(),
        "regenerate_count": regen,
        "_deps": {"settings": settings},
        "warnings": [],
    }


def test_valid_grounded_answer_passes():
    ans = (
        "Under color of law a person who deprives another of constitutional "
        "rights shall be liable to the party injured [S1]. A municipality may "
        "not assert qualified immunity under section 1983 [S2]."
    )
    out = validate_answer(_state(ans))
    assert out["valid"] is True
    assert all(c.verified for c in out["citations"])


def test_unresolved_marker_fails():
    out = validate_answer(_state("The city is always liable [S9]."))
    assert out["valid"] is False
    assert any(not c.verified for c in out["citations"])


def test_uncited_factual_sentence_triggers_regenerate():
    # Factual claim with no marker -> invalid, and under budget -> regenerate.
    out = validate_answer(_state("Municipalities enjoy absolute immunity always."))
    assert out["valid"] is False
    assert out.get("strict") is True
    assert out["regenerate_count"] == 1


def test_abstention_is_valid():
    out = validate_answer(_state("I cannot answer this question from the provided sources."))
    assert out["valid"] is True
    assert out["abstained"] is True


def test_ungrounded_citation_is_downgraded():
    # Marker resolves, but the sentence talks about something the passage lacks.
    ans = "Patent term extensions last twenty years for pharmaceutical drugs [S1]."
    out = validate_answer(_state(ans))
    # Either flagged invalid or the citation marked unverified.
    assert out["valid"] is False or any(not c.verified for c in out["citations"])


def test_exhausted_budget_routes_to_human_review():
    settings = Settings.from_env()
    state = {
        "answer": "Unsupported claim with no citation whatsoever.",
        "context_map": _ctx(),
        "regenerate_count": settings.max_regenerate,  # already at the cap
        "_deps": {"settings": settings},
        "warnings": [],
    }
    out = validate_answer(state)
    assert out["valid"] is False
    assert out["needs_human_review"] is True


class _BadThenGoodLLM:
    """First answer is ungrounded; after STRICT_MODE it grounds correctly.

    The 'good' answer delegates to the real MockLLM so it cites whatever
    markers the live context actually assigned (marker numbering is dynamic).
    """

    def __init__(self, settings: Settings):
        self.calls = 0
        self._good = MockLLM(settings)

    def generate(self, system: str, prompt: str) -> str:
        self.calls += 1
        if "STRICT_MODE" in system:
            return self._good.generate(system, prompt)
        return "Cities are entirely immune from all lawsuits everywhere."


def test_regenerate_loop_recovers(store):
    """Drive the real sequential runner with an LLM that fails then succeeds."""
    settings = Settings.from_env()
    runner = SequentialRunner(store, settings)
    state = {"query": "Does a municipality get qualified immunity under section 1983?"}
    # Inject deps with our scripted LLM.
    from app.retrieval.reranker import get_reranker

    bad_llm = _BadThenGoodLLM(settings)
    state["_deps"] = {
        "store": store,
        "settings": settings,
        "llm": bad_llm,
        "reranker": get_reranker(settings),
    }
    out = runner.invoke(state)
    assert bad_llm.calls >= 2          # regenerated at least once
    assert out["regenerate_count"] >= 1
    assert out["valid"] is True        # recovered to a grounded answer
