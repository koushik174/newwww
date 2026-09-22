"""Tests for the query router (corpus classification)."""

from __future__ import annotations

from app.graph.nodes.route import classify
from app.schemas import SourceType


def test_statute_query_routes_to_federal():
    routed = classify("What does 42 U.S.C. 1983 provide?")
    assert SourceType.FEDERAL in routed


def test_regulation_deadline_query_routes_to_cfr():
    routed = classify("How many days do I have to file an EEOC charge deadline?")
    assert SourceType.CFR in routed


def test_case_query_routes_to_caselaw():
    routed = classify("Was Monroe v. Pape overruled by a later Supreme Court case?")
    assert SourceType.CASELAW in routed


def test_cross_domain_query_routes_to_multiple():
    routed = classify(
        "Can a city be sued under the 1983 statute and what did the Monell case hold?"
    )
    assert SourceType.FEDERAL in routed
    assert SourceType.CASELAW in routed


def test_unclear_query_falls_back_to_all_corpora():
    routed = classify("tell me about legal stuff")
    assert set(routed) == {SourceType.FEDERAL, SourceType.CFR, SourceType.CASELAW}


def test_routing_is_authority_ordered():
    routed = classify(
        "1983 statute, its CFR regulation deadline, and the controlling court case"
    )
    ranks = [s.authority_rank for s in routed]
    assert ranks == sorted(ranks)
