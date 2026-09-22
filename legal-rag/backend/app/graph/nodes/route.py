"""Node 2: router — classify the query to one or more corpora.

WHY route at all: retrieving from all three corpora for every query wastes
compute and, worse, dilutes the candidate pool with off-target passages that
can crowd out the right answer after fusion. A statute question ("what does 42
U.S.C. 1983 provide?") should not be answered primarily from case law. The
router is intentionally *recall-oriented*: when signals are weak or the query
spans domains (e.g. a filing-deadline question that touches both a statute and
its implementing regulation) it selects multiple corpora rather than gambling
on one. This is rule-based and transparent here; the production swap is an LLM
or small classifier behind the same ``route`` signature.
"""

from __future__ import annotations

import re

from ...schemas import SourceType
from ..state import GraphState

# Signal keywords per corpus. Deliberately broad to favor recall.
_FEDERAL_SIGNALS = [
    r"\bu\.?s\.?c\.?\b", r"\bstatute", r"\bsection \d+", r"\b\d+ u\.?s\.?c",
    r"\bcivil rights\b", r"\b1983\b", r"\b2000e", r"\bcongress",
    r"\btitle vii\b", r"\bemployment\b", r"\bdiscriminat", r"\bunlawful\b",
    r"\bstatutory\b",
]
_CFR_SIGNALS = [
    r"\bc\.?f\.?r\.?\b", r"\bregulation", r"\bregulatory", r"\brule\b",
    r"\bdeadline", r"\bfil(?:e|ing|ed)\b", r"\beeoc\b", r"\bagency\b",
    r"\bhow many days\b", r"\b\d+ days\b", r"\bcharge\b", r"\blimitation",
    r"\bdeferral\b",
]
_CASELAW_SIGNALS = [
    r"\bcase\b", r"\bcourt\b", r"\b v\.? \b", r"\boverrul", r"\bprecedent",
    r"\bheld\b", r"\bopinion\b", r"\bsupreme court\b", r"\bmonell\b", r"\bowen\b",
    r"\bimmunity\b", r"\bmunicipal", r"\bcity\b", r"\bcities\b", r"\bsued?\b",
    r"\bliab", r"\bperson\b", r"\bpolicy\b", r"\bcustom\b", r"\brespondeat\b",
]

_TABLE = {
    SourceType.FEDERAL: [re.compile(p, re.I) for p in _FEDERAL_SIGNALS],
    SourceType.CFR: [re.compile(p, re.I) for p in _CFR_SIGNALS],
    SourceType.CASELAW: [re.compile(p, re.I) for p in _CASELAW_SIGNALS],
}


def classify(query: str) -> list[SourceType]:
    """Return every corpus with at least one signal; fall back to all three."""
    hits: list[SourceType] = []
    for source_type, patterns in _TABLE.items():
        if any(p.search(query) for p in patterns):
            hits.append(source_type)
    if not hits:
        # No strong signal: search everything (recall over precision).
        hits = [SourceType.FEDERAL, SourceType.CFR, SourceType.CASELAW]
    # Keep canonical authority order for deterministic downstream behavior.
    hits.sort(key=lambda s: s.authority_rank)
    return hits


def route(state: GraphState) -> GraphState:
    query = state.get("sanitized_query") or state.get("query", "")
    return {"routed_to": classify(query)}
