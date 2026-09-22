"""Node 7: generation.

WHY the prompt is this strict: the single biggest failure mode of legal RAG is a
fluent, confident answer that is not actually supported by the sources. So the
system prompt constrains the model to answer ONLY from the numbered context,
cite every factual sentence with a ``[S#]`` marker, address every detected
conflict, and abstain when the context does not support an answer. The user
prompt carries the context block and the question in a fixed, parseable layout
(``QUESTION:`` line, ``[S#]`` passages) so both the mock and real providers, and
the validator, can rely on its structure.

On a regenerate pass ``strict`` is set: we escalate the system prompt (marked
``STRICT_MODE``) to demand tighter grounding, which is what makes the
regenerate loop corrective rather than merely repetitive.
"""

from __future__ import annotations

from ...llm import LLM
from ..state import GraphState

_BASE_SYSTEM = (
    "You are a meticulous US legal research assistant. Follow these rules "
    "without exception:\n"
    "1. Answer ONLY using the numbered SOURCES provided. Never use outside "
    "knowledge.\n"
    "2. Every factual sentence MUST end with one or more citation markers like "
    "[S1] or [S2] naming the source(s) that support it.\n"
    "3. If the sources do not support an answer, explicitly say you cannot "
    "answer from the provided sources. Do not guess.\n"
    "4. If DETECTED CONFLICTS are listed, address each one explicitly and "
    "explain which source controls and why.\n"
    "5. Be concise and precise; do not invent citations."
)

_STRICT_SUFFIX = (
    "\nSTRICT_MODE: A previous answer failed validation. Cite EVERY sentence, "
    "drop any claim you cannot tie to a specific source, and prefer abstaining "
    "over an unsupported statement."
)


def generate(state: GraphState) -> GraphState:
    deps = state["_deps"]
    llm: LLM = deps["llm"]
    strict = bool(state.get("strict"))

    system = _BASE_SYSTEM + (_STRICT_SUFFIX if strict else "")
    question = state.get("sanitized_query") or state.get("query", "")
    prompt = f"{state.get('context_text', '')}\n\nQUESTION: {question}\n\nANSWER:"

    answer = llm.generate(system, prompt).strip()
    return {"answer": answer}
