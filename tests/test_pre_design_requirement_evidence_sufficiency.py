from __future__ import annotations

import pytest

from minecraft_mod_ai import authored_scope_research_contract as authored_scope
from minecraft_mod_ai import pre_design_rag_quality_contract as quality

_PROMPT = "우주선 조종과 행성 스캔 기능을 모두 구현해"
_PILOT_QUERIES = [
    "Minecraft vehicle control entity architecture",
    "Fabric player controlled entity input",
]
_SCAN_QUERIES = [
    "Minecraft world scan server architecture",
    "Fabric server side area query pattern",
]


def _catalog():
    return {
        "requirements": [
            {
                "requirement_id": "req:pilot",
                "search_queries": list(_PILOT_QUERIES),
            },
            {
                "requirement_id": "req:scan",
                "search_queries": list(_SCAN_QUERIES),
            },
        ]
    }


def _domain(*, queries=None):
    return {
        "domain_id": "request",
        "requirements": [_PROMPT],
        "queries": list(queries or [*_PILOT_QUERIES, *_SCAN_QUERIES]),
    }


def _row(query: str, *, content: str = ""):
    records = []
    if content:
        records.append(
            {
                "source_id": "project:test:" + query.replace(" ", "_"),
                "source_type": "project_source",
                "content": content,
                "retrieval_section": "project_rag",
            }
        )
    return {
        "query": query,
        "evidence_records": records,
        "retrieval_errors": [],
    }


def _bind_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        authored_scope,
        "_active_catalog",
        lambda prompt: _catalog() if prompt == _PROMPT else None,
    )


def test_sibling_evidence_cannot_mask_uncovered_approved_requirement(monkeypatch) -> None:
    _bind_catalog(monkeypatch)
    grounded = {
        "queries": [
            _row(_PILOT_QUERIES[0], content="Entity control is handled on the server."),
            _row(_PILOT_QUERIES[1]),
            _row(_SCAN_QUERIES[0]),
            _row(_SCAN_QUERIES[1]),
        ]
    }

    with pytest.raises(ValueError, match="req:scan"):
        quality.fuse_grounded_domain_evidence(_domain(), grounded)


def test_each_requirement_needs_one_content_bearing_approved_query(monkeypatch) -> None:
    _bind_catalog(monkeypatch)
    grounded = {
        "queries": [
            _row(_PILOT_QUERIES[0], content="Entity control is handled on the server."),
            _row(_PILOT_QUERIES[1]),
            _row(_SCAN_QUERIES[0]),
            _row(_SCAN_QUERIES[1], content="Area scans run against server world state."),
        ]
    }

    fused = quality.fuse_grounded_domain_evidence(_domain(), grounded)

    receipt = fused["requirement_sufficiency"]
    assert receipt["authority"] == "approved_requirement_retrieval_plan"
    assert receipt["required_requirement_count"] == 2
    assert receipt["satisfied_requirement_count"] == 2
    assert receipt["unresolved_requirement_ids"] == []
    assert receipt["sufficient"] is True
    by_id = {item["requirement_id"]: item for item in receipt["requirements"]}
    assert by_id["req:pilot"]["queries"] == _PILOT_QUERIES
    assert by_id["req:pilot"]["queries_with_content"] == [_PILOT_QUERIES[0]]
    assert by_id["req:scan"]["queries"] == _SCAN_QUERIES
    assert by_id["req:scan"]["queries_with_content"] == [_SCAN_QUERIES[1]]
    assert fused["fusion"]["queries_with_content"] == 2


def test_flat_query_list_cannot_drift_from_approved_requirement_plan(monkeypatch) -> None:
    _bind_catalog(monkeypatch)
    incomplete_domain = _domain(
        queries=[*_PILOT_QUERIES, _SCAN_QUERIES[0]],
    )

    with pytest.raises(ValueError, match="provenance drift.*req:scan"):
        quality._approved_requirement_query_obligations(incomplete_domain)
