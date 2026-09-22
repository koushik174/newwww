"""Documented loaders for the real, public-domain US legal corpora.

IMPORTANT: nothing here runs at build/import time and nothing is scraped
automatically. This module DOCUMENTS, with authoritative URLs, how to pull each
corpus in bulk and how to normalize it into the :class:`Chunk` contract used by
the rest of the system. Run the functions explicitly (they raise a clear
``NotImplementedError`` describing the manual step) when you are ready to ingest
real data. US government works (statutes, regulations, judicial opinions) are
not subject to federal copyright, but always review each source's terms and rate
limits before bulk downloading.

Corpora and official bulk sources
---------------------------------
1. U.S. Code (Federal statutes)
   Office of the Law Revision Counsel (OLRC) bulk XML (USLM format):
     https://uscode.house.gov/download/download.shtml
   Each title is a zipped USLM XML file; parse <section> elements into chunks,
   keeping the citation (e.g. "42 U.S.C. § 1983"), title heading, and text.

2. Code of Federal Regulations (CFR)
   GovInfo bulk data (also USLM/XML), from the Government Publishing Office:
     https://www.govinfo.gov/bulkdata/CFR
   API: https://api.govinfo.gov/  (register for a free api.data.gov key)
   Parse <SECTION> nodes; keep the part/section number and the effective date.

3. Case law
   Caselaw Access Project (Harvard LIL) — bulk/download:
     https://case.law/  /  https://static.case.law/
   CourtListener (Free Law Project) REST API + bulk data:
     https://www.courtlistener.com/help/api/rest/
   Parse opinions into chunks; capture the reporter citation, court, year, and
   any subsequent-history flags (overruled/superseded) into ``metadata``.

Chunking guidance
-----------------
Legal text has natural units (section, subsection, paragraph). Prefer semantic
chunking on those boundaries over fixed token windows so a citation always
points at a coherent, quotable unit — and record ``page``/``paragraph`` (and,
once ingesting PDFs, bounding-box coordinates) into each :class:`Chunk` so the
citation UI can deep-link to the exact location.
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import Chunk, SourceType

USC_BULK_URL = "https://uscode.house.gov/download/download.shtml"
CFR_BULK_URL = "https://www.govinfo.gov/bulkdata/CFR"
GOVINFO_API = "https://api.govinfo.gov/"
CAP_URL = "https://case.law/"
COURTLISTENER_API = "https://www.courtlistener.com/help/api/rest/"


def fetch_us_code(titles: list[int], out_dir: Path) -> list[Chunk]:
    """Download + parse the given U.S. Code titles into chunks.

    Implementation outline:
      1. GET the per-title USLM XML zip from ``USC_BULK_URL``.
      2. Unzip and stream-parse <section> elements (lxml).
      3. For each section emit a Chunk(source_type=FEDERAL, citation="N U.S.C.
         § M", title=<heading>, text=<normalized text>, paragraph=<subsec id>).
    """
    raise NotImplementedError(
        "Bulk ingestion is intentionally manual. Download USLM XML from "
        f"{USC_BULK_URL} and parse <section> elements into Chunk objects."
    )


def fetch_cfr(titles: list[int], out_dir: Path) -> list[Chunk]:
    """Download + parse CFR titles from GovInfo bulk data into chunks."""
    raise NotImplementedError(
        "Download CFR bulk XML from "
        f"{CFR_BULK_URL} (or use {GOVINFO_API} with an api.data.gov key) and "
        "parse <SECTION> nodes into Chunk objects."
    )


def fetch_caselaw(query: str, out_dir: Path) -> list[Chunk]:
    """Fetch opinions from CAP or CourtListener into chunks.

    Capture subsequent history into metadata, e.g.
    ``metadata={"status": "overruled", "overruled_by": "<cite>"}`` so the
    conflict builder can flag stale holdings.
    """
    raise NotImplementedError(
        "Use the Caselaw Access Project bulk data at "
        f"{CAP_URL} or the CourtListener REST API at {COURTLISTENER_API} to "
        "download opinions, then chunk into Chunk objects with citation "
        "metadata."
    )


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
