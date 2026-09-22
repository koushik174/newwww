"""Node 5: source merger + conflict builder.

WHY merge with an authority order: once we have the top passages we present them
to the model in order of legal authority — statute (U.S. Code) > regulation
(CFR) > case law — because that is the order a lawyer weighs them, and it biases
the model toward resolving tension in favor of the controlling source.
Deduplication removes the same passage arriving from two corpora/retrievers.

WHY a dedicated conflict node: legal sources routinely disagree, and a RAG
system that silently averages them is dangerous. Rather than hide disagreement,
we *detect and surface* it so the model must confront it. We look for three
concrete, checkable signals:

  1. Differing deadlines / day-counts for the same kind of action (e.g. a
     180-day vs a 300-day filing window) — the classic statute/reg tension.
  2. Mandatory vs permissive language ("shall" vs "may") on the same obligation.
  3. Superseded / overruled markers carried in a passage's metadata.

Detected conflicts flow into the context node, which injects them with an
explicit instruction to resolve them in the answer.
"""

from __future__ import annotations

import re

from ...schemas import Chunk, Conflict, SourceType
from ..state import GraphState

_DAYS_RE = re.compile(r"(\d{1,4})\s*[- ]?\s*day", re.I)
_FILING_CONTEXT = re.compile(
    r"\b(fil|charge|complaint|claim|deadline|limitation|within)\b", re.I
)


def merge_sources(chunks: list[Chunk]) -> list[Chunk]:
    """Dedupe by chunk key and order by legal authority, then original order."""
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in chunks:
        if c.key in seen:
            continue
        seen.add(c.key)
        deduped.append(c)
    # Stable sort by authority rank; preserves rerank order within a tier.
    deduped.sort(key=lambda c: c.source_type.authority_rank)
    return deduped


def _marker_map(chunks: list[Chunk]) -> dict[str, str]:
    """Map each chunk key to its [S#] marker for the (already ordered) list."""
    return {c.key: f"S{i + 1}" for i, c in enumerate(chunks)}


def build_conflicts(chunks: list[Chunk]) -> list[Conflict]:
    """Detect disagreements across the merged passages."""
    conflicts: list[Conflict] = []
    markers = _marker_map(chunks)

    # (1) Differing filing deadlines / day-counts.
    deadline_hits: list[tuple[str, int, Chunk]] = []
    for c in chunks:
        if _FILING_CONTEXT.search(c.text):
            for m in _DAYS_RE.finditer(c.text):
                deadline_hits.append((markers[c.key], int(m.group(1)), c))
    distinct_days = {d for _, d, _ in deadline_hits}
    if len(distinct_days) >= 2:
        involved = sorted({mk for mk, _, _ in deadline_hits})
        day_by_marker = {mk: d for mk, d, _ in deadline_hits}
        conflicts.append(
            Conflict(
                kind="deadline",
                description=(
                    "Sources state different filing deadlines: "
                    + ", ".join(
                        f"[{mk}] = {day_by_marker[mk]} days" for mk in involved
                    )
                    + ". The applicable window depends on jurisdiction/agency "
                    "and which source controls."
                ),
                markers=[f"[{mk}]" for mk in involved],
                details={"days": sorted(distinct_days)},
            )
        )

    # (2) Mandatory vs permissive language on filing obligations.
    mandatory = [markers[c.key] for c in chunks
                 if _FILING_CONTEXT.search(c.text) and re.search(r"\bshall\b|\bmust\b", c.text, re.I)]
    permissive = [markers[c.key] for c in chunks
                  if _FILING_CONTEXT.search(c.text) and re.search(r"\bmay\b", c.text, re.I)]
    if mandatory and permissive and set(mandatory) != set(permissive):
        conflicts.append(
            Conflict(
                kind="mandatory_vs_permissive",
                description=(
                    "Sources differ on whether the obligation is mandatory "
                    "('shall'/'must') or permissive ('may')."
                ),
                markers=[f"[{m}]" for m in sorted(set(mandatory) | set(permissive))],
                details={"mandatory": mandatory, "permissive": permissive},
            )
        )

    # (3) Superseded / overruled markers from metadata.
    for c in chunks:
        status = str(c.metadata.get("status", "")).lower()
        if status in {"overruled", "superseded", "abrogated"}:
            ref = c.metadata.get("superseded_by") or c.metadata.get("overruled_by")
            conflicts.append(
                Conflict(
                    kind="overruled" if status == "overruled" else "superseded",
                    description=(
                        f"[{markers[c.key]}] ({c.title}) is marked {status}"
                        + (f" by {ref}" if ref else "")
                        + "; treat its holding with caution."
                    ),
                    markers=[f"[{markers[c.key]}]"],
                    details={"status": status, "reference": ref},
                )
            )

    return conflicts


def merge_conflict(state: GraphState) -> GraphState:
    reranked = state.get("reranked", [])
    merged = merge_sources(reranked)
    conflicts = build_conflicts(merged)
    return {"merged": merged, "conflicts": conflicts}
