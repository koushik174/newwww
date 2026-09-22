"""FastAPI application: async /query and /health.

WHY startup index build: retrieval indexes are built once during the app
lifespan and stored on ``app.state``, so every request reuses the same in-memory
BM25/vector structures instead of paying ingestion cost per call. The heavy
pipeline is CPU-bound and synchronous, so /query offloads ``run_pipeline`` to a
worker thread to keep the event loop responsive under concurrency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .embeddings import get_embedder
from .graph.build_graph import engine_name, run_pipeline
from .ingest.build_indexes import build_store
from .llm import llm_mode
from .schemas import HealthResponse, QueryRequest, QueryResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    embedder = get_embedder(settings)
    app.state.settings = settings
    app.state.store = build_store(settings, embedder=embedder)
    app.state.embedding_mode = (
        "real" if settings.use_real_embeddings else "mock"
    )
    yield
    # No teardown needed for the in-memory store.


app = FastAPI(
    title="Legal RAG API",
    version="0.1.0",
    description="Grounded, citation-validated answers over US legal corpora.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = app.state.settings
    store = app.state.store
    return HealthResponse(
        status="ok",
        corpora=store.counts(),
        llm_mode=llm_mode(settings),
        embedding_mode=app.state.embedding_mode,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    settings = app.state.settings
    store = app.state.store
    # Pipeline is synchronous/CPU-bound; run off the event loop.
    return await run_in_threadpool(
        run_pipeline, req.query, store, settings, req.top_k
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "legal-rag",
        "engine": engine_name(),
        "docs": "/docs",
        "endpoints": ["POST /query", "GET /health"],
    }
