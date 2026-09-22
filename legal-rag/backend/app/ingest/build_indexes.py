"""Load seed corpora from JSON and build a :class:`CorpusStore`.

WHY a build step: retrieval indexes are constructed once (at API startup, or at
the top of the demo/eval) from the on-disk corpus, then shared read-only across
requests. Keeping ingestion separate from serving means we can later point the
same builder at real bulk corpora (see fetch_public_data.py) without touching
the graph or API code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from ..embeddings import Embedder, get_embedder
from ..retrieval.indexes import CorpusStore
from ..schemas import Chunk, SourceType

_FILES = {
    SourceType.FEDERAL: "federal/seed.json",
    SourceType.CFR: "cfr/seed.json",
    SourceType.CASELAW: "caselaw/seed.json",
}


def load_chunks(data_dir: Path, source_type: SourceType) -> list[Chunk]:
    """Read one corpus's seed.json into validated :class:`Chunk` objects."""
    path = data_dir / _FILES[source_type]
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for item in raw:
        item.setdefault("source_type", source_type.value)
        chunks.append(Chunk(**item))
    return chunks


def build_store(
    settings: Optional[Settings] = None, embedder: Optional[Embedder] = None
) -> CorpusStore:
    """Build the in-memory store for all three corpora."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    store = CorpusStore(settings, embedder=embedder)
    for source_type in _FILES:
        chunks = load_chunks(settings.data_dir, source_type)
        store.add_corpus(source_type, chunks)
    return store


if __name__ == "__main__":  # pragma: no cover
    store = build_store()
    print("Built corpus store:", store.counts())
