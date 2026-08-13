from __future__ import annotations

import httpx

from minecraft_mod_ai.ecosystem_discovery import EcosystemDiscoveryClient
from minecraft_mod_ai.planning_stall_guard_contract import _planning_seed_brief
from minecraft_mod_ai.research_coordinator import collect_ecosystem_seed_bundle
import minecraft_mod_ai.planning_stall_guard_contract as stall_guard


def _brief() -> dict:
    return {
        "schema_version": "mmm/central-research-brief-v1",
        "brief_sha256": "sha256:root-cause",
        "domains": [
            {
                "domain_id": "mechanics",
                "objective": "mechanics",
                "requirements": ["r"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["q1", "q2", "q3"],
                "providers": ["official_docs", "modrinth", "github"],
                "depends_on": [],
            }
        ],
    }


def test_planner_seed_does_not_exhaust_route_catalog() -> None:
    calls: list[str] = []

    def page_builder(
        prompt: str,
        game_design: dict,
        *,
        research_brief: dict | None,
        client,
        route_cursor: str,
        route_limit: int,
    ) -> dict:
        del prompt, game_design, research_brief, client
        calls.append(route_cursor)
        assert route_limit == 2
        if route_cursor:
            raise AssertionError("planner seed must not request a second route page")
        return {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "status": "empty",
            "query_sha256": "sha256:q",
            "route_sha256": "sha256:r",
            "route_count": 4,
            "route_offset": 0,
            "processed_route_count": 2,
            "remaining_route_count": 2,
            "next_route_cursor": "next-page",
            "routes_complete": False,
            "candidate_count": 0,
            "pages": [],
            "errors": [],
            "coverage": "seed",
            "authorization": "none",
            "download_performed": False,
        }

    result = collect_ecosystem_seed_bundle(
        "prompt",
        {},
        research_brief=_brief(),
        route_limit=2,
        page_builder=page_builder,
        allow_legacy_terminal=True,
    )

    assert calls == [""]
    assert result["routes_complete"] is False
    assert result["remaining_route_count"] == 2
    assert result["next_route_cursor"] == "next-page"
    assert result["collection_receipt"]["planning_seed_only"] is True


def test_exhaustive_specialist_collector_still_continues_route_pages() -> None:
    calls: list[str] = []

    def page_builder(
        prompt: str,
        game_design: dict,
        *,
        research_brief: dict | None,
        client,
        route_cursor: str,
        route_limit: int,
    ) -> dict:
        del prompt, game_design, research_brief, client
        calls.append(route_cursor)
        first = not route_cursor
        return {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "status": "empty",
            "query_sha256": "sha256:q",
            "route_sha256": "sha256:r",
            "route_count": 4,
            "route_offset": 0 if first else 2,
            "processed_route_count": 2,
            "remaining_route_count": 2 if first else 0,
            "next_route_cursor": "next-page" if first else "",
            "routes_complete": not first,
            "candidate_count": 0,
            "pages": [],
            "errors": [],
            "coverage": "seed",
            "authorization": "none",
            "download_performed": False,
        }

    result = collect_ecosystem_seed_bundle(
        "prompt",
        {},
        research_brief=_brief(),
        route_limit=2,
        page_builder=page_builder,
    )

    assert calls == ["", "next-page"]
    assert result["routes_complete"] is True
    assert result["remaining_route_count"] == 0


def test_stall_guard_does_not_own_a_second_research_executor() -> None:
    assert not hasattr(stall_guard, "_ECOSYSTEM_EXECUTOR")
    assert not hasattr(stall_guard, "_STATE")


def test_discovery_client_reuses_one_http_pool_across_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    client = EcosystemDiscoveryClient(transport=httpx.MockTransport(handler))
    pooled = client._mmm_http_client
    first = client._get_json("https://api.modrinth.com/v2/search")
    second = client._get_json("https://api.modrinth.com/v2/search")

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert client._mmm_http_client is pooled
    assert len(requests) == 2


def test_normal_planner_seed_does_not_need_query_compaction_to_terminate(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN", raising=False)
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_ROUTE_BUDGET", raising=False)
    brief = _brief()
    projected = _planning_seed_brief(brief)

    assert projected["domains"][0]["queries"] == brief["domains"][0]["queries"]
    assert projected["_mmm_planning_seed_projection"]["compacted"] is False
