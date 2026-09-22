"""Shared pytest fixtures. Adds backend/ to sys.path and builds the store once."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings  # noqa: E402
from app.ingest.build_indexes import build_store  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def store(settings):
    # Offline by default (mock LLM + mock embedder); no downloads.
    return build_store(settings)
