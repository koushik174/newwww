"""Shared text utilities: legal-aware sentence splitting and term extraction.

WHY this lives in one place: the generator (mock LLM) and the validator MUST
agree on what a "sentence" is and what a "content term" is, or grounding checks
will disagree with generation. Legal prose is full of periods that are not
sentence boundaries ("42 U.S.C. § 1983", "Monell v. Dep't of Soc. Servs., 436
U.S. 658"), so a naive ``split('.')`` shatters a single citation into fake
"uncited sentences". This module protects common legal abbreviations before
splitting.
"""

from __future__ import annotations

import re

# Abbreviations whose trailing period must NOT end a sentence. Longest first so
# "U.S.C." is protected before "U.S.".
_ABBREVIATIONS = [
    "U.S.C.", "U.S.", "C.F.R.", "e.g.", "i.e.", "cf.", "No.", "Nos.",
    "Dep't", "Servs.", "Inc.", "Co.", "Corp.", "Ass'n", "Cir.", "Ct.",
    "Art.", "Sec.", "S.Ct.", "F.2d", "F.3d", "F. Supp.", "v.", "vs.",
    "pp.", "p.", "al.", "Fed.", "Reg.",
]
_PLACEHOLDER = "\x00"

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'(§0-9])")
_WORD_RE = re.compile(r"[a-z0-9]+")
_MARKER_RE = re.compile(r"\[S\d+\]")

# Function words and pipeline meta-vocabulary that should not count as
# "content" when measuring grounding overlap.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "on",
    "with", "that", "this", "be", "by", "under", "it", "as", "from", "at",
    "was", "were", "has", "have", "not", "no", "any", "such", "which", "who",
    "whom", "shall", "may", "must", "will", "would", "can", "could", "into",
    "within", "after", "before", "than", "then", "there", "their", "its",
    "his", "her", "they", "them", "these", "those", "other", "otherwise",
    "according", "source", "sources", "controls", "control", "controlling",
    "authoritative", "more", "less", "compare", "note", "among", "because",
    "when", "where", "whether", "also", "each", "over", "per", "about",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking on legal abbreviations.

    Newlines are treated as hard sentence boundaries (the mock LLM emits one
    grounded claim per line); within a line, an abbreviation-aware regex finds
    the real sentence breaks.
    """
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))

    sentences: list[str] = []
    for line in protected.split("\n"):
        line = line.strip()
        if not line:
            continue
        for part in _SENT_BOUNDARY.split(line):
            restored = part.replace(_PLACEHOLDER, ".").strip()
            if restored:
                sentences.append(restored)
    return sentences


def first_sentence(text: str, max_chars: int = 400) -> str:
    """Return the first (abbreviation-aware) sentence, capped in length."""
    text = " ".join(text.split())
    sents = split_sentences(text)
    clause = sents[0] if sents else text
    if len(clause) > max_chars:
        clause = clause[:max_chars].rsplit(" ", 1)[0] + "..."
    return clause


def content_terms(text: str) -> set[str]:
    """Lowercased, de-markered, de-stopworded terms used for overlap scoring."""
    text = _MARKER_RE.sub(" ", text)
    return {w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}
