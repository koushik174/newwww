"""Node 8: answer validation (the trust gate).

WHY this node is the heart of the system: an answer is only returnable if it is
verifiably grounded. We run three independent checks:

  (a) RESOLUTION — every ``[S#]`` marker in the answer must resolve to a passage
      that was actually retrieved and shown to the model. A marker pointing at a
      non-existent source is an immediate fail (a fabricated citation).
  (b) GROUNDING — for each cited factual sentence, the cited passage(s) must
      actually support it. Offline we approximate support with content-term
      coverage between the sentence and the union of the passages it cites; the
      production swap is an NLI/LLM entailment check behind this same function.
      A summary/conflict sentence that only references markers already grounded
      by their own source sentences is accepted (it introduces no new claim).
  (c) NO UNCITED FACTUAL SENTENCES — every substantive sentence must carry at
      least one marker. Abstention/meta sentences are exempt.

If any check fails and we are under ``MAX_REGENERATE``, we set ``strict`` and
loop back to generate with a harder prompt. If we exhaust the budget, we return
``valid=false`` and flag the response for the human review queue — the system
refuses to emit an ungrounded legal answer.
"""

from __future__ import annotations

import re

from ...config import Settings
from ...schemas import Chunk, Citation, SourceType
from ...text_utils import content_terms, split_sentences
from ..state import GraphState

_MARKER_RE = re.compile(r"\[S(\d+)\]")
_ABSTAIN_RE = re.compile(
    r"cannot answer|declin\w+ to answer|do not have|don't have|no .*information|"
    r"not supported by|unable to answer",
    re.I,
)

# Minimum share of a sentence's content terms that must appear in the cited
# passage(s) for the sentence to count as directly grounded.
_GROUNDING_THRESHOLD = 0.5
# A sentence with fewer content terms than this is treated as non-substantive
# (meta/transition) and is exempt from the citation requirement.
_MIN_FACTUAL_TERMS = 2


def _markers_in(text: str) -> list[str]:
    return [f"S{n}" for n in _MARKER_RE.findall(text)]


def _is_abstention(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text))


def validate_answer(state: GraphState) -> GraphState:
    deps = state["_deps"]
    settings: Settings = deps["settings"]
    answer = state.get("answer", "")
    context_map: dict[str, Chunk] = state.get("context_map", {})
    regenerate_count = int(state.get("regenerate_count", 0))
    warnings = list(state.get("warnings", []))

    abstained = _is_abstention(answer) and not _MARKER_RE.search(answer)
    problems: list[str] = []

    # --- (a) resolution: build citations, flag unresolved markers ----------
    default_type = (
        next(iter(context_map.values())).source_type if context_map else SourceType.FEDERAL
    )
    citations: list[Citation] = []
    for marker in dict.fromkeys(_markers_in(answer)):  # unique, order-preserving
        chunk = context_map.get(marker)
        if chunk is None:
            citations.append(
                Citation(
                    marker=marker,
                    source_id="<unresolved>",
                    source_type=default_type,
                    title="<unresolved>",
                    verified=False,
                    reason="Marker does not resolve to any retrieved passage.",
                )
            )
            problems.append(f"[{marker}] does not resolve to a retrieved passage.")
        else:
            citations.append(
                Citation(
                    marker=marker,
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    page=chunk.page,
                    paragraph=chunk.paragraph,
                    verified=False,  # upgraded once grounded below
                )
            )

    # --- (b)+(c) per-sentence grounding and uncited-factual checks ---------
    sentences = split_sentences(answer)
    grounded_markers: set[str] = set()
    deferred: list[tuple[str, list[str]]] = []  # sentences to resolve in pass 2

    for sent in sentences:
        if _is_abstention(sent):
            continue
        markers = [m for m in _markers_in(sent) if m in context_map]
        sent_terms = content_terms(sent)
        factual = len(sent_terms) >= _MIN_FACTUAL_TERMS

        if factual and not markers:
            problems.append(f"Uncited factual sentence: {sent[:80]!r}")
            continue
        if not markers:
            continue  # non-factual, uncited -> fine (transition/meta)

        # Direct grounding: union coverage of the cited passages.
        passage_terms: set[str] = set()
        for m in markers:
            ch = context_map[m]
            passage_terms |= content_terms(ch.text + " " + ch.title)
        coverage = (len(sent_terms & passage_terms) / len(sent_terms)) if sent_terms else 1.0

        if coverage >= _GROUNDING_THRESHOLD:
            grounded_markers.update(markers)
        else:
            # Might still be a summary of already-grounded sources; decide later.
            deferred.append((sent, markers))

    # Pass 2: a deferred sentence is acceptable iff every marker it cites was
    # grounded by some other (source) sentence — it makes no new claim.
    for sent, markers in deferred:
        if all(m in grounded_markers for m in markers):
            continue
        problems.append(
            f"Sentence not supported by its citations: {sent[:80]!r}"
        )

    # Upgrade citations whose marker was grounded.
    for cit in citations:
        if cit.source_id == "<unresolved>":
            continue
        if cit.marker in grounded_markers:
            cit.verified = True
        else:
            cit.reason = cit.reason or "Cited passage does not support the sentence."

    valid = (len(problems) == 0 and (bool(grounded_markers) or abstained)) or abstained

    if valid:
        return {
            "valid": True,
            "abstained": abstained,
            "citations": citations,
            "warnings": warnings,
            "needs_human_review": False,
        }

    if regenerate_count < settings.max_regenerate:
        warnings.append(
            f"Validation failed (attempt {regenerate_count + 1}): "
            + "; ".join(problems[:3])
        )
        return {
            "valid": False,
            "citations": citations,
            "regenerate_count": regenerate_count + 1,
            "strict": True,
            "warnings": warnings,
            "abstained": abstained,
        }

    warnings.append(
        "Validation failed after max regenerations; routing to human review: "
        + "; ".join(problems[:3])
    )
    return {
        "valid": False,
        "citations": citations,
        "needs_human_review": True,
        "warnings": warnings,
        "abstained": abstained,
    }


def should_regenerate(state: GraphState) -> str:
    """Conditional edge: 'regenerate' loops back to generate, 'end' finishes."""
    if state.get("valid"):
        return "end"
    if state.get("needs_human_review"):
        return "end"
    if state.get("strict"):
        return "regenerate"
    return "end"
