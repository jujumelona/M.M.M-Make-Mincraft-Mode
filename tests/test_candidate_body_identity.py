from __future__ import annotations

from minecraft_mod_ai.planning_candidate_evidence import (
    global_grounded_pool,
    requirement_candidate_trace,
    semantic_frontier_pool,
)


def _requirement() -> dict[str, object]:
    return {
        "decision_type": "requirement",
        "requirement_id": "req_001",
        "semantic_capability": "spacecraft upgrade",
        "statement": "Players upgrade spacecraft.",
        "acceptance": ["Players can upgrade spacecraft."],
    }


def _grounded(domain_id: str) -> dict[str, object]:
    return {
        "queries": [
            {
                "query": "spacecraft",
                "origin_domain_id": domain_id,
                "evidence_records": [
                    {
                        "source_id": "modrinth:same-project",
                        "content": "Spacecraft documentation only.",
                    }
                ],
            },
            {
                "query": "upgrade",
                "origin_domain_id": domain_id,
                "evidence_records": [
                    {
                        "source_id": "modrinth:same-project",
                        "content": "Upgrade documentation only.",
                    }
                ],
            },
        ]
    }


def test_changed_source_body_versions_cannot_fuse_lexical_facets() -> None:
    pool = global_grounded_pool({"r_002": _grounded("r_002")})
    trace = requirement_candidate_trace(_requirement(), pool)

    assert len(trace["candidates"]) == 2
    assert len({row["content_sha256"] for row in trace["candidates"]}) == 2
    assert all(len(row["matched_facets"]) == 1 for row in trace["candidates"])
    assert trace["lexical_coverage_complete"] is False
    assert trace["missing_facets"] == [["spacecraft"], ["upgrade"]]

    frontier = semantic_frontier_pool(_requirement(), pool, trace)
    assert frontier["queries"] == []


def test_direct_route_keeps_body_specific_partial_candidates_for_semantic_review() -> None:
    pool = global_grounded_pool({"r_001": _grounded("r_001")})
    trace = requirement_candidate_trace(_requirement(), pool)
    frontier = semantic_frontier_pool(_requirement(), pool, trace)

    assert trace["lexical_coverage_complete"] is False
    records = [
        record
        for query in frontier["queries"]
        for record in query["evidence_records"]
    ]
    assert [record["content"] for record in records] == [
        "Spacecraft documentation only.",
        "Upgrade documentation only.",
    ]
