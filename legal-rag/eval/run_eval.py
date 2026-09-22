"""Offline evaluation harness + promotion gate.

WHY eval-as-a-gate: a legal RAG system should not ship a change that quietly
degrades grounding. This script runs a golden set through the full pipeline and
computes three metrics, then prints a PASS/FAIL "promotion gate" that a CI job
can key on (non-zero exit on FAIL). The same idea scales to a real regression
suite gating deploys.

Metrics
-------
* retrieval_recall  : fraction of expected source ids that appear in the merged
                      top-k passages (did we retrieve the right law?).
* citation_accuracy : fraction of emitted citations that are verified (resolve +
                      grounded) (did we cite honestly?).
* grounded_rate     : fraction of queries whose answer passed validation, i.e.
                      is fully grounded or a clean abstention (is the answer
                      trustworthy?).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.config import get_settings  # noqa: E402
from app.graph.build_graph import run_pipeline  # noqa: E402
from app.ingest.build_indexes import build_store  # noqa: E402

# Promotion thresholds. Tune per risk tolerance; CI fails below these.
GATE = {
    "retrieval_recall": 0.80,
    "citation_accuracy": 0.90,
    "grounded_rate": 1.00,
}


def load_golden(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def run() -> int:
    settings = get_settings()
    store = build_store(settings)
    golden_path = Path(__file__).resolve().parent / "golden.jsonl"
    golden = load_golden(golden_path)

    recall_scores: list[float] = []
    citation_scores: list[float] = []
    grounded_flags: list[bool] = []

    print("=" * 78)
    print("LEGAL RAG — EVALUATION")
    print("=" * 78)

    for item in golden:
        resp = run_pipeline(item["query"], store, settings)
        # Recall is measured against the sources the model actually grounded on
        # (its citations), which is the meaningful end-to-end retrieval signal:
        # a passage that was retrieved but never cited did not help the answer.
        retrieved_ids = {cit.source_id for cit in resp.citations}
        expected = set(item.get("expected_sources", []))
        recall = (
            len(expected & retrieved_ids) / len(expected) if expected else 1.0
        )
        recall_scores.append(recall)

        verified = [c for c in resp.citations if c.verified]
        citation_acc = (
            len(verified) / len(resp.citations) if resp.citations else 1.0
        )
        citation_scores.append(citation_acc)

        grounded_flags.append(bool(resp.valid))

        must = [m.lower() for m in item.get("must_mention", [])]
        mention_hits = sum(1 for m in must if m in resp.answer.lower())
        print(
            f"\n[{item['id']}] {item['query']}"
            f"\n   routed_to        : {[s.value for s in resp.routed_to]}"
            f"\n   recall           : {recall:.2f} "
            f"(expected {sorted(expected)} | cited {sorted(retrieved_ids)})"
            f"\n   citation_accuracy: {citation_acc:.2f} "
            f"({len(verified)}/{len(resp.citations)} verified)"
            f"\n   valid            : {resp.valid} (regenerated {resp.regenerated})"
            f"\n   must_mention     : {mention_hits}/{len(must)} "
            f"{[m for m in must if m in resp.answer.lower()]}"
            f"\n   conflicts        : {[c.kind for c in resp.conflicts]}"
        )

    def _avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "retrieval_recall": _avg(recall_scores),
        "citation_accuracy": _avg(citation_scores),
        "grounded_rate": _avg([1.0 if g else 0.0 for g in grounded_flags]),
    }

    print("\n" + "=" * 78)
    print("AGGREGATE METRICS")
    print("=" * 78)
    passed = True
    for name, value in metrics.items():
        threshold = GATE[name]
        ok = value >= threshold
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(f"   {name:18s}: {value:.3f}  (gate >= {threshold:.2f})  [{status}]")

    print("=" * 78)
    if passed:
        print("PROMOTION GATE: PASS ✅  — safe to promote.")
    else:
        print("PROMOTION GATE: FAIL ❌  — do NOT promote.")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(run())
