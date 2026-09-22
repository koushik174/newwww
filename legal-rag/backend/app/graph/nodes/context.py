"""Node 6: context builder.

WHY numbered [S#] passages with provenance: grounding is only auditable if every
passage the model sees has a stable, resolvable handle. We number the merged
sources ``[S1]..[Sn]`` and carry each one's title, citation, page and paragraph
into the context block, and we keep a ``context_map`` ({"S1": Chunk}) so the
answer's ``[S#]`` markers can be parsed straight back to source objects for
validation and for clickable frontend citations.

The passage text is emitted on a single line beginning with the marker, with an
indented ``META:`` line for provenance. This layout is unambiguous to parse (the
model, mock or real, reads the text; the ``META:`` line never leaks into the
answer) and keeps citations resolvable.

WHY inject conflicts here: detected disagreements are written into the context
as ``CONFLICT:`` lines with an explicit instruction to resolve them, so the
model cannot paper over a 180-vs-300-day discrepancy — it must address it.
"""

from __future__ import annotations

from ...schemas import Chunk, Conflict
from ..state import GraphState


def _format_passage(marker: str, chunk: Chunk) -> str:
    text = " ".join(chunk.text.split())  # single line, collapsed whitespace
    prov = []
    if chunk.page is not None:
        prov.append(f"p.{chunk.page}")
    if chunk.paragraph:
        prov.append(f"¶{chunk.paragraph}")
    meta = " | ".join(
        [chunk.title, chunk.citation or "", " ".join(prov)]
    ).strip(" |")
    return f"[{marker}] {text}\n     META: {meta}"


def build_context_text(
    chunks: list[Chunk], conflicts: list[Conflict]
) -> tuple[str, dict[str, Chunk]]:
    context_map: dict[str, Chunk] = {}
    lines: list[str] = []
    for i, chunk in enumerate(chunks):
        marker = f"S{i + 1}"
        context_map[marker] = chunk
        lines.append(_format_passage(marker, chunk))

    parts = ["SOURCES:", *lines]
    if conflicts:
        parts.append("")
        parts.append("DETECTED CONFLICTS (you MUST address each in your answer):")
        for conf in conflicts:
            parts.append(f"CONFLICT: {conf.description}")
    return "\n".join(parts), context_map


def build_context(state: GraphState) -> GraphState:
    merged = state.get("merged", [])
    conflicts = state.get("conflicts", [])
    context_text, context_map = build_context_text(merged, conflicts)
    return {"context_text": context_text, "context_map": context_map}
