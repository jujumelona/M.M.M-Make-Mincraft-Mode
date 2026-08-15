from __future__ import annotations

import json

import pytest

import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _implementation_prompt,
)
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.platform_resolver import resolve_platform, retarget_proposal
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
    capability: str = "ai_inference",
) -> dict:
    return {
        "schema_version": "mmm/technology-radar-page-v1",
        "source_sha256": "sha256:" + "a" * 64,
        "target": {"minecraft_version": "1.20.1"},
        "target_evidence_policy": {"official": True},
        "classification": {"ai_requested": True},
        "voice_contract": {"activated": False},
        "requirements": [
            {
                "requirement_id": f"technology_{index:04d}",
                "domain_id": f"domain_{index:04d}",
                "capability_kind": capability,
                "required_gates": ["exact_compatibility"],
                "required_tests": ["runtime"],
                "deterministic_fallback": "Disable safely.",
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
        "discovery_policy": {"authorization": "none"},
        "scale_policy": "Continue until empty.",
        "radar_sha256": "sha256:" + f"{offset:064x}",
    }


def _ecosystem_page(
    *,
    offset: int,
    advanced_to: int,
    total: int,
    next_cursor: str,
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
    assert aggregate["aggregate_schema_version"] == (
        "mmm/technology-radar-aggregate-v1"
    )
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
            "Build technology",
            page_size=2,
            page_builder=builder,
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
                offset=0,
                advanced_to=advanced_to,
                total=3,
                next_cursor="again",
            )
        return _ecosystem_page(
            offset=1,
            advanced_to=2,
            total=3,
            next_cursor="again",
        )

    match = "did not advance" if failure == "non_advancing" else "repeated a cursor"
    with pytest.raises(SpecValidationError, match=match):
        collect_ecosystem_seed_bundle(
            "Build systems",
            {"title": "Test"},
            page_builder=builder,
        )


def test_planner_sidecar_uses_capability_from_later_technology_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = HeuristicPlanner().plan("Create one technology anchor item")
    base = retarget_proposal(
        base,
        resolve_platform("Minecraft 1.20.1 Fabric"),
    )
    game_design = {
        "title": "Paged technology",
        "pitch": "Use every classified capability.",
        "_research_brief": {"domains": []},
    }
    monkeypatch.setattr(
        planner_module.GameDesignPlanner,
        "plan",
        lambda self, value, media_paths=(): (game_design, base),
    )
    monkeypatch.setattr(
        planner_module,
        "_retrieve_implementation_evidence",
        lambda *args, **kwargs: {"schema_version": "test/evidence-v1"},
    )
    monkeypatch.setattr(
        planner_module,
        "discover_seed_bundle",
        lambda *args, **kwargs: _ecosystem_page(
            offset=0, advanced_to=0, total=0, next_cursor=""
        ),
    )
    calls: list[str] = []

    def technology_builder(prompt, research_brief, *, cursor, page_size, **kwargs):
        del prompt, research_brief, kwargs
        calls.append(cursor)
        if not cursor:
            return _technology_page(
                offset=0,
                returned=50,
                total=51,
                page_size=page_size,
                next_cursor="technology:50",
                capability="voice_transport",
            )
        page = _technology_page(
            offset=50,
            returned=1,
            total=51,
            page_size=page_size,
            next_cursor="",
            capability="speech_synthesis",
        )
        page["requirements"][0]["requirement_id"] = "late_speech_synthesis"
        return page

    monkeypatch.setattr(
        planner_module,
        "build_technology_radar",
        technology_builder,
    )

    class Router:
        prompt = ""

        def generate_text(self, role, messages, **kwargs):
            del role, kwargs
            self.prompt = messages[-1]["content"]
            return json.dumps(
                {
                    "modules": [],
                    "assets": [],
                    "audio": [],
                    "acceptance_tests": ["The technology has a safe fallback."],
                }
            )

    router = Router()
    proposal = CompleteGameDesignPlanner(router).plan("Build every capability")

    assert calls == ["", "technology:50"]
    assert len(proposal.game_design["_technology_radar"]["requirements"]) == 51
    sidecar = next(
        module
        for module in proposal.modules
        if module.config.get("integration_type") == "mmm_local_ai_sidecar"
    )
    assert sidecar.config["capabilities"] == ["speech_synthesis"]
    assert '"requirement_count": 51' in router.prompt
    assert "late_speech_synthesis" in router.prompt


def test_actual_planning_prompt_is_bounded_but_reports_full_aggregate_counts() -> None:
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
    rendered = _implementation_prompt(
        "Build every requested large-system capability.",
        {
            "title": "Large",
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
        },
    )
    encoded_context = rendered.split(
        "Compact authoritative planning context:\n", 1
    )[1].split("\n\nCreate only the paginated production outline.", 1)[0]
    view = json.loads(encoded_context)["research_outline"]

    technology = view["technology_radar"]
    ecosystem = view["ecosystem"]
    assert technology["requirement_count"] == 500
    assert technology["capability_counts"] == {"ai_inference": 500}
    assert len(technology["requirements"]) == 1
    assert technology["requirement_view_complete"] is False
    assert technology["requirements_receipt"]["byte_length"] > 0
    assert len(technology["requirements_receipt"]["sha256"]) == 64
    assert ecosystem["route_count"] == 500
    assert ecosystem["representative_candidate_count"] == 1
    assert "github:owner/repo-0" in rendered
    assert "github:owner/repo-499" not in rendered
    assert len(rendered.encode("utf-8")) < 12_000
