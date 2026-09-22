"""Node 1: input validation + prompt-injection normalization.

WHY this node: user input in a legal setting is both a correctness and a
security surface. Before any retrieval we (a) enforce the Pydantic contract
(non-empty, bounded length), (b) normalize the text (collapse control
characters and whitespace) so downstream tokenization is stable, and (c) scan
for prompt-injection patterns ("ignore previous instructions", "you are now",
attempts to change the system role). We do NOT silently drop the query on a
suspected injection — we strip the offending directive, keep the substantive
question, and record a warning, because over-blocking real legal questions is
its own failure mode.
"""

from __future__ import annotations

import re

from ..state import GraphState

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all|any|the)?\s*(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard (?:all|any|the)?\s*(?:previous|prior|above)", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"reveal (?:your )?(?:system )?(?:prompt|instructions)", re.I),
    re.compile(r"act as (?:an?|the) (?:developer|admin|root)", re.I),
]


def validate_input(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    warnings = list(state.get("warnings", []))
    errors = list(state.get("errors", []))

    if not query:
        errors.append("Empty query.")
        return {"errors": errors, "sanitized_query": ""}

    # Normalize: strip control chars, collapse whitespace.
    cleaned = _CONTROL_RE.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Detect and neutralize injection directives (strip the sentence, keep rest).
    flagged = False
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            flagged = True
            cleaned = pat.sub(" ", cleaned)
    if flagged:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        warnings.append(
            "Possible prompt-injection detected and neutralized in the query."
        )

    if not cleaned:
        # The query was *only* an injection attempt.
        errors.append("Query contained no substantive question after sanitization.")

    return {
        "sanitized_query": cleaned,
        "warnings": warnings,
        "errors": errors,
    }
