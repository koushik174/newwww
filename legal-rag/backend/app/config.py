"""Centralized, environment-driven configuration for the Legal RAG pipeline.

WHY a single settings object: every knob that changes retrieval behavior
(top-N per retriever, RRF constant, rerank depth, regenerate budget, model
names, backend selection) lives in one place so the graph nodes stay pure and
testable. Nodes receive a ``Settings`` instance rather than reading os.environ
directly, which makes it trivial to run the same nodes under different configs
in tests and in the eval harness.

This module deliberately uses only the standard library (a dataclass reading
``os.environ``) so that ``import app.config`` never fails when optional
dependencies such as ``pydantic-settings`` are absent. The whole system must be
importable and runnable offline with zero third-party packages installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Project root: backend/app/config.py -> parents[2] == the legal-rag/ repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Runtime configuration. Construct via :meth:`from_env`.

    All retrieval and generation tunables are surfaced here so operators can
    reshape the pipeline without touching code, and so tests can pin exact
    values for deterministic assertions.
    """

    # --- Retrieval tunables -------------------------------------------------
    bm25_top_n: int = 20          # lexical candidates per corpus before fusion
    vector_top_n: int = 20        # dense candidates per corpus before fusion
    rrf_k: int = 60               # RRF smoothing constant (see rrf.py for WHY 60)
    rerank_top_k: int = 6         # passages kept after cross-encoder rerank
    max_regenerate: int = 2       # regeneration attempts before human review

    # --- Model / provider selection ----------------------------------------
    llm_provider: str = "anthropic"                       # anthropic | openai
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Feature gates ------------------------------------------------------
    # Real network/model calls are OFF by default so the system runs offline,
    # deterministically, and with zero downloads. Flip these to opt in.
    use_real_llm: bool = False
    use_real_embeddings: bool = False
    use_real_reranker: bool = False

    # --- Vector backend -----------------------------------------------------
    vector_backend: str = "numpy"                         # numpy | qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "legal_rag"
    embedding_dim: int = 256      # dimension for the offline mock embedder

    # --- Data ---------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _REPO_ROOT / "data")

    # --- Secrets (read but never logged) ------------------------------------
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables, falling back to defaults."""
        data_dir_env = os.environ.get("DATA_DIR")
        return cls(
            bm25_top_n=_get_int("BM25_TOP_N", 20),
            vector_top_n=_get_int("VECTOR_TOP_N", 20),
            rrf_k=_get_int("RRF_K", 60),
            rerank_top_k=_get_int("RERANK_TOP_K", 6),
            max_regenerate=_get_int("MAX_REGENERATE", 2),
            llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
            anthropic_model=os.environ.get(
                "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
            ),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            reranker_model=os.environ.get(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            use_real_llm=_get_bool("USE_REAL_LLM", False),
            use_real_embeddings=_get_bool("USE_REAL_EMBEDDINGS", False),
            use_real_reranker=_get_bool("USE_REAL_RERANKER", False),
            vector_backend=os.environ.get("VECTOR_BACKEND", "numpy"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection_prefix=os.environ.get(
                "QDRANT_COLLECTION_PREFIX", "legal_rag"
            ),
            embedding_dim=_get_int("EMBEDDING_DIM", 256),
            data_dir=Path(data_dir_env) if data_dir_env else _REPO_ROOT / "data",
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        )


def get_settings() -> Settings:
    """Convenience accessor used by the API layer and demo."""
    return Settings.from_env()
