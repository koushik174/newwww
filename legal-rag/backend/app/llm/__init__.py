"""Provider-agnostic LLM wrapper with a deterministic offline mock.

WHY a wrapper: the graph's generate node should not know or care whether it is
talking to Anthropic, OpenAI, or a mock. Every provider implements one method::

    generate(system: str, prompt: str) -> str

Real providers are gated behind ``USE_REAL_LLM=1`` and read their keys from the
environment. When no key is present, or the flag is off, we fall back to a
``MockLLM`` that answers *only* from the provided numbered context, cites every
factual sentence with a ``[S#]`` marker, addresses injected conflicts, and
abstains when nothing supports an answer. The mock is deterministic so tests,
the demo and CI produce identical output with zero network calls.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..config import Settings
from ..text_utils import content_terms, first_sentence

# Passage lines look like: "[S1] <text>"  (META lines are ignored).
_PASSAGE_RE = re.compile(r"^\[S(\d+)\]\s+(.*)$")
_CONFLICT_RE = re.compile(r"^CONFLICT:\s*(.+)$")
_MARKER_RE = re.compile(r"\[S\d+\]")


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str:
        ...


class MockLLM:
    """Deterministic, context-grounded answer synthesizer.

    Strategy (mirrors what we prompt a real model to do):
      1. Parse the numbered ``[S#]`` passages out of the prompt's context.
      2. Score each passage by content-term overlap with the user question.
      3. Emit one grounded sentence per relevant passage: the passage's own
         leading sentence, tagged with its ``[S#]`` marker, so the downstream
         grounding check always passes for supported claims.
      4. If conflicts were injected, add a sentence that names the conflict and
         cites the markers involved — satisfying "address conflicts explicitly"
         while only referencing sources already grounded above.
      5. If no passage is relevant, abstain with a non-factual sentence, which
         the validator treats as a valid abstention.

    Under a "stricter" regenerate prompt (STRICT_MODE in the system message) it
    raises the relevance threshold and drops weakly supported sentences,
    demonstrating that the regenerate loop changes behavior rather than
    repeating itself.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, system: str, prompt: str) -> str:
        strict = "STRICT_MODE" in system
        q_match = re.search(r"QUESTION:\s*(.+)", prompt)
        question = q_match.group(1).strip() if q_match else prompt
        q_terms = content_terms(question)

        passages: list[tuple[int, str]] = []
        conflicts: list[str] = []
        for line in prompt.splitlines():
            pm = _PASSAGE_RE.match(line.strip())
            if pm:
                passages.append((int(pm.group(1)), pm.group(2).strip()))
                continue
            cm = _CONFLICT_RE.match(line.strip())
            if cm:
                conflicts.append(cm.group(1).strip())

        scored = [
            (len(q_terms & content_terms(body)), num, body)
            for num, body in passages
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))

        threshold = 2 if strict else 1
        chosen = [(num, body) for overlap, num, body in scored if overlap >= threshold]
        if not chosen and not strict:
            chosen = [(num, body) for overlap, num, body in scored[:1] if overlap >= 1]

        if not chosen:
            return (
                "I cannot answer this question from the provided sources, so I "
                "am declining to answer rather than speculate."
            )

        chosen_markers = {f"S{num}" for num, _ in chosen}
        sentences: list[str] = []
        for num, body in chosen:
            clause = first_sentence(body).rstrip(" .;:")
            sentences.append(f"{clause} [S{num}].")

        # Address each conflict, citing only markers that are already grounded.
        for conflict in conflicts:
            markers = [m for m in _MARKER_RE.findall(conflict)
                       if m.strip("[]") in chosen_markers]
            if not markers:
                continue
            marker_str = " ".join(markers)
            sentences.append(
                f"The provided sources conflict and the more authoritative one "
                f"governs {marker_str}."
            )

        return "\n".join(sentences)


class AnthropicLLM:  # pragma: no cover - requires network + key
    """Anthropic Claude provider (default when USE_REAL_LLM=1)."""

    def __init__(self, settings: Settings) -> None:
        import anthropic  # type: ignore

        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def generate(self, system: str, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAILLM:  # pragma: no cover - requires network + key
    """OpenAI provider (optional)."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def generate(self, system: str, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


def get_llm(settings: Settings) -> LLM:
    """Factory: real provider only when enabled AND a key is present."""
    if settings.use_real_llm:
        try:
            if settings.llm_provider == "openai" and settings.openai_api_key:
                return OpenAILLM(settings)
            if settings.anthropic_api_key:
                return AnthropicLLM(settings)
        except Exception:  # pragma: no cover
            pass
    return MockLLM(settings)


def llm_mode(settings: Settings) -> str:
    if settings.use_real_llm and (settings.anthropic_api_key or settings.openai_api_key):
        return f"real:{settings.llm_provider}"
    return "mock"
