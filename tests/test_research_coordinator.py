from __future__ import annotations

import pytest

from minecraft_mod_ai.complete_planner import _implementation_research_outline
from minecraft_mod_ai.research_coordinator import (
    collect_ecosystem_seed_bundle,
    collect_technology_radar,
)
from minecraft_mod_ai.spec import SpecValidationError


def _technology_page(
    *,
    offset: int,
    returned: int,
    total: int,
    page_size: int,
    next_cursor: str,
) -> dict:
    return {
        "schema_version": "mmm/technology-radar-page-v1",
        "source_sha256": "sha256:" + "a" * 64,
        "requirements": [
            {
                "requirement_id": f"requirement_{index:06d}",
                "domain_id": f"domain_{index:06d}",
                "capability_kind": "ai_inference",
            }
            for index in range(offset, offset + returned)
        ],
        "pagination": {
            "offset": offset,
            "page_size": page_size,
            "returned": returned,
            "total_requirements": total,
            "next_cursor": next_cursor,
        },
        "radar_sha256": f"page-{offset}",
    }


def _ecosystem_page(
    *, offset: int, advanced_to: int, total: int, next_cursor: str
) -> dict:
    returned = advanced_to - offset
    return {
        "schema_version": "mmm/ecosystem-seed-bundle-v1",
        "status": "available",
        "query_sha256": "sha256:" + "b" * 64,
        "route_sha256": "sha256:" + "c" * 64,
        "route_count": total,
        "route_offset": offset,
        "processed_route_count": returned,
        "remaining_route_count": total - advanced_to,
        "next_route_cursor": next_cursor,
        "routes_complete": not next_cursor,
        "candidate_count": returned,
        "pages": [
            {
                "research_domain_id": f"domain_{index:04d}",
                "provider": "github",
                "returned": 1,
                "next_cursor": "",
                "page_sha256": f"page-{index}",
                "candidates": [],
            }
            for index in range(offset, advanced_to)
        ],
        "errors": [],
        "coverage": "seed pages",
        "authorization": "none",
        "download_performed": False,
    }


def test_technology_coordinator_exhausts_large_catalog_without_global_cap() -> None:
    total = 451
    page_size = 3
    calls: list[str] = []

    def builder(prompt, research_brief, *, cursor, page_size, **kwargs):
        del prompt, research_brief, kwargs
        calls.append(cursor)
        offset = int(cursor.split(":", 1)[1]) if cursor else 0
        returned = min(page_size, total - offset)
        advanced_to = offset + returned
        next_cursor = f"cursor:{advanced_to}" if advanced_to < total else ""
        return _technology_page(
            offset=offset,
            returned=returned,
            total=total,
            page_size=page_size,
            next_cursor=next_cursor,
        )

    aggregate = collect_technology_radar(
        "Build every requested technology",
        {"domains": []},
        page_size=page_size,
        page_builder=builder,
    )
    assert len(calls) == 151
    assert len(aggregate["requirements"]) == total
    assert aggregate["pagination"] == {
        "offset": 0,
        "page_size": page_size,
        "returned": total,
        "total_requirements": total,
        "next_cursor": "",
        "pages_collected": 151,
        "complete": True,
    }
    assert aggregate["aggregate_schema_version"] == "mmm/technology-radar-aggregate-v1"
    assert aggregate["collection_receipt"]["page_count"] == 151


@pytest.mark.parametrize("failure", ["repeated", "non_advancing"])
def test_technology_coordinator_rejects_bad_cursor_progress(failure: str) -> None:
    def builder(prompt, research_brief, *, cursor, page_size, **kwargs):
        del prompt, research_brief, kwargs
        if not cursor:
            returned = 0 if failure == "non_advancing" else 1
            return _technology_page(
                offset=0,
                returned=returned,
                total=3,
                page_size=page_size,
                next_cursor="again",
            )
        return _technology_page(
            offset=1,
            returned=1,
            total=3,
            page_size=page_size,
            next_cursor="again",
        )

    match = "did not advance" if failure == "non_advancing" else "repeated a cursor"
    with pytest.raises(SpecValidationError, match=match):
        collect_technology_radar(
            "Build technology", page_size=2, page_builder=builder
        )


def test_ecosystem_coordinator_exhausts_every_route_page() -> None:
    total = 233
    route_limit = 2
    calls: list[str] = []

    def builder(prompt, game_design, *, route_cursor, route_limit, **kwargs):
        del prompt, game_design, kwargs
        calls.append(route_cursor)
        offset = int(route_cursor.split(":", 1)[1]) if route_cursor else 0
        advanced_to = min(total, offset + route_limit)
        next_cursor = f"route:{advanced_to}" if advanced_to < total else ""
        return _ecosystem_page(
            offset=offset,
            advanced_to=advanced_to,
            total=total,
            next_cursor=next_cursor,
        )

    aggregate = collect_ecosystem_seed_bundle(
        "Build every requested system",
        {"title": "Large"},
        route_limit=route_limit,
        page_builder=builder,
    )
    assert len(calls) == 117
    assert aggregate["routes_complete"] is True
    assert aggregate["next_route_cursor"] == ""
    assert aggregate["route_count"] == total
    assert aggregate["processed_route_count"] == total
    assert aggregate["remaining_route_count"] == 0
    assert len(aggregate["pages"]) == total
    assert aggregate["candidate_count"] == total
    assert aggregate["collection_receipt"]["route_page_count"] == 117


def test_ecosystem_coordinator_exhausts_route_cursor_when_discovery_is_disabled() -> None:
    total = 23
    route_limit = 5
    calls: list[str] = []

    def builder(prompt, game_design, *, route_cursor, route_limit, **kwargs):
        del prompt, game_design, kwargs
        calls.append(route_cursor)
        offset = int(route_cursor.split(":", 1)[1]) if route_cursor else 0
        advanced_to = min(total, offset + route_limit)
        next_cursor = f"route:{advanced_to}" if advanced_to < total else ""
        return {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "status": "disabled",
            "query_sha256": "sha256:" + "b" * 64,
            "route_sha256": "sha256:" + "c" * 64,
            "route_count": total,
            "route_offset": offset,
            "processed_route_count": 0,
            "remaining_route_count": total - offset,
            "next_route_cursor": next_cursor,
            "routes_complete": not next_cursor,
            "candidate_count": 0,
            "pages": [],
            "errors": [],
        }

    aggregate = collect_ecosystem_seed_bundle(
        "Build systems",
        {"title": "Disabled"},
        route_limit=route_limit,
        page_builder=builder,
    )
    assert len(calls) == 5
    assert aggregate["routes_complete"] is True
    assert aggregate["status"] == "disabled"
    assert aggregate["processed_route_count"] == 0
    assert aggregate["remaining_route_count"] == total


def test_real_ecosystem_route_cursor_is_exhausted_in_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = {
        "domains": [
            {
                "domain_id": f"system_{index:03d}",
                "objective": f"Research system {index}",
                "requirements": [f"Implement system {index}"],
                "evidence_kinds": ["source_code"],
                "queries": [f"system {index}"],
                "providers": ["github"],
                "depends_on": [],
            }
            for index in range(23)
        ]
    }
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
    aggregate = collect_ecosystem_seed_bundle(
        "Build all systems",
        {"title": "Offline"},
        research_brief=brief,
        route_limit=5,
    )
    assert aggregate["route_count"] == 23
    assert aggregate["routes_complete"] is True
    assert aggregate["collection_receipt"]["route_page_count"] == 5
    assert aggregate["status"] == "disabled"


@pytest.mark.parametrize("failure", ["repeated", "non_advancing"])
def test_ecosystem_coordinator_rejects_bad_cursor_progress(failure: str) -> None:
    def builder(prompt, game_design, *, route_cursor, **kwargs):
        del prompt, game_design, kwargs
        if not route_cursor:
            advanced_to = 0 if failure == "non_advancing" else 1
            return _ecosystem_page(
                offset=0, advanced_to=advanced_to, total=3, next_cursor="again"
            )
        return _ecosystem_page(
            offset=1, advanced_to=2, total=3, next_cursor="again"
        )

    match = "did not advance" if failure == "non_advancing" else "repeated a cursor"
    with pytest.raises(SpecValidationError, match=match):
        collect_ecosystem_seed_bundle(
            "Build systems", {"title": "Test"}, page_builder=builder
        )


def test_implementation_outline_excludes_old_unbounded_aggregate_contracts() -> None:
    requirements = [
        {
            "requirement_id": f"same_kind_{index}",
            "domain_id": f"domain_{index}",
            "capability_kind": "ai_inference",
        }
        for index in range(500)
    ]
    pages = [
        {
            "research_domain_id": f"domain_{index}",
            "provider": "github",
            "returned": 1,
            "candidates": [{"candidate_id": f"github:owner/repo-{index}"}],
        }
        for index in range(500)
    ]
    outline = _implementation_research_outline(
        {
            "mod_id": "large_system",
            "description": "Build every requested large-system capability.",
            "_technology_radar": {
                "requirements": requirements,
                "pagination": {"total_requirements": 500},
            },
            "_ecosystem_discovery": {
                "route_count": 500,
                "candidate_count": 500,
                "pages": pages,
                "errors": [],
            },
        }
    )

    assert outline == {
        "mod_id": "large_system",
        "description": "Build every requested large-system capability.",
    }
    rendered = repr(outline)
    assert "same_kind_0" not in rendered
    assert "same_kind_499" not in rendered
    assert "github:owner/repo-0" not in rendered
    assert "github:owner/repo-499" not in rendered
    assert len(rendered.encode("utf-8")) < 4000
