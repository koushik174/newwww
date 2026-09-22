"""Pydantic v2 I/O contracts for the Legal RAG system.

WHY Pydantic v2 everywhere: legal answers are only as trustworthy as their
provenance. By making ``Chunk``, ``Citation`` and ``Conflict`` first-class,
validated models, every stage of the pipeline exchanges typed objects rather
than loose dicts, so a passage can never lose the ``page``/``paragraph``
provenance that makes a citation verifiable. The same models serialize
straight to the API response, giving the frontend clickable, resolvable
citations for free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """The three corpora this system reasons over.

    Ordered by legal authority for the source-merger (statute > regulation >
    case law): a controlling statute outranks an implementing regulation,
    which in turn outranks persuasive/interpretive case law.
    """

    FEDERAL = "federal"   # U.S. Code (statutes)
    CFR = "cfr"           # Code of Federal Regulations
    CASELAW = "caselaw"   # judicial opinions

    @property
    def authority_rank(self) -> int:
        """Lower is more authoritative; used to order merged sources."""
        return {"federal": 0, "cfr": 1, "caselaw": 2}[self.value]


class Chunk(BaseModel):
    """A retrievable passage carrying full provenance.

    ``page`` and ``paragraph`` are optional today but are the hooks for
    coordinate/bounding-box citation later: once documents are ingested from
    PDFs we can attach bbox metadata without changing this contract.
    """

    source_id: str = Field(..., description="Stable id, e.g. '42-usc-1983'.")
    source_type: SourceType
    title: str = Field(..., description="Human-readable citation title.")
    text: str
    page: Optional[int] = None
    paragraph: Optional[str] = None
    citation: Optional[str] = Field(
        default=None, description="Bluebook-style citation string."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Unique identity for dedup/fusion (source + paragraph anchor)."""
        return f"{self.source_id}#{self.paragraph or 'p'}"


class ScoredChunk(BaseModel):
    """A chunk paired with a retrieval/fusion/rerank score."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A ``[S#]`` marker resolved back to its source passage plus validation."""

    marker: str = Field(..., description="The in-text marker, e.g. 'S1'.")
    source_id: str
    source_type: SourceType
    title: str
    page: Optional[int] = None
    paragraph: Optional[str] = None
    verified: bool = Field(
        default=False,
        description="True once the marker resolves AND grounding is confirmed.",
    )
    reason: Optional[str] = Field(
        default=None, description="Why verification failed, if it did."
    )


class Conflict(BaseModel):
    """A detected disagreement across sources, surfaced to the LLM to resolve."""

    kind: str = Field(
        ...,
        description="deadline | mandatory_vs_permissive | superseded | overruled",
    )
    description: str
    markers: list[str] = Field(
        default_factory=list, description="Context markers ([S#]) in conflict."
    )
    details: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Inbound API contract."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override RERANK_TOP_K."
    )


class QueryResponse(BaseModel):
    """Outbound API contract: the answer plus everything needed to trust it."""

    query: str
    routed_to: list[SourceType]
    answer: str
    citations: list[Citation]
    conflicts: list[Conflict] = Field(default_factory=list)
    valid: bool
    abstained: bool = False
    regenerated: int = Field(
        default=0, description="How many regeneration passes were used."
    )
    candidate_count: int = 0
    top_k_count: int = 0
    needs_human_review: bool = False
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    corpora: dict[str, int]
    llm_mode: str
    embedding_mode: str
