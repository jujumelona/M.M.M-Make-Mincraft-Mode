from __future__ import annotations

import minecraft_mod_ai.complete_planner as complete_planner
import minecraft_mod_ai.parallel_runtime_contract as parallel_runtime
from minecraft_mod_ai.ecosystem_discovery import EcosystemDiscoveryClient
from minecraft_mod_ai.planning_stall_guard_contract import (
    _ecosystem_key,
    _planning_seed_brief,
)


def _brief() -> dict:
    return {
        "schema_version": "mmm/central-research-brief-v1",
        "brief_sha256": "sha256:test",
        "domains": [
            {
                "domain_id": "gameplay",
                "objective": "gameplay",
                "requirements": ["a"],
                "evidence_kinds": ["minecraft_api"],
                "queries": ["q1", "q2", "q3"],
                "providers": ["official_docs", "modrinth", "github"],
                "depends_on": [],
            },
            {
                "domain_id": "visuals",
                "objective": "visuals",
                "requirements": ["b"],
                "evidence_kinds": ["visual_reference"],
                "queries": ["v1", "v2"],
                "providers": ["openverse_images"],
                "depends_on": [],
            },
        ],
    }


def test_normal_seed_projection_is_lossless(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN", raising=False)
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_ROUTE_BUDGET", raising=False)
    original = _brief()
    projected = _planning_seed_brief(original)

    assert [item["domain_id"] for item in projected["domains"]] == ["gameplay", "visuals"]
    assert projected["domains"][0]["providers"] == original["domains"][0]["providers"]
    assert projected["domains"][1]["providers"] == original["domains"][1]["providers"]
    assert projected["domains"][0]["queries"] == ["q1", "q2", "q3"]
    assert projected["domains"][1]["queries"] == ["v1", "v2"]
    projection = projected["_mmm_planning_seed_projection"]
    assert projection["schema_version"] == "mmm/planning-seed-projection-v3"
    assert projection["compacted"] is False
    assert projection["reason"] == "lossless_route_graph_external_io_deferred"
    assert projection["specialist_discovery_continues_full_brief"] is True


def test_legacy_seed_query_limit_does_not_cut_deferred_route_graph(monkeypatch) -> None:
    monkeypatch.setenv("MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN", "1")
    original = _brief()
    projected = _planning_seed_brief(original)

    assert projected["domains"][0]["queries"] == ["q1", "q2", "q3"]
    assert projected["domains"][1]["queries"] == ["v1", "v2"]
    assert original["domains"][0]["queries"] == ["q1", "q2", "q3"]
    projection = projected["_mmm_planning_seed_projection"]
    assert projection["compacted"] is False
    assert projection["queries_per_domain"] is None
    assert projection["full_research_brief_retained"] is True


def test_large_route_fanout_is_lossless_when_network_io_is_deferred(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN", raising=False)
    monkeypatch.setenv("MMM_ECOSYSTEM_SEED_ROUTE_BUDGET", "16")
    original = _brief()
    original["domains"][0]["queries"] = [f"q{i}" for i in range(20)]

    projected = _planning_seed_brief(original)

    projection = projected["_mmm_planning_seed_projection"]
    assert projection["route_budget"] == 16
    assert projection["estimated_external_routes"] > 16
    assert projection["compacted"] is False
    assert projection["reason"] == "lossless_route_graph_external_io_deferred"
    assert projected["domains"][0]["queries"] == original["domains"][0]["queries"]
    assert projected["domains"][0]["providers"] == original["domains"][0]["providers"]
    assert projection["specialist_discovery_continues_full_brief"] is True


def test_ecosystem_key_reuses_effective_platform_target() -> None:
    game_design = {
        "_platform_selection": {
            "target": {
                "minecraft_version": "1.20.1",
                "loader": "fabric",
                "mappings": "yarn-1.20.1+build.1",
            }
        }
    }
    original = _brief()
    targeted = {
        **original,
        "_mmm_platform_target": dict(game_design["_platform_selection"]["target"]),
    }
    page_builder = object()

    assert _ecosystem_key("prompt", game_design, original, page_builder) == _ecosystem_key(
        "prompt",
        game_design,
        targeted,
        page_builder,
    )

    other_target = {
        **original,
        "_mmm_platform_target": {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
        },
    }
    assert _ecosystem_key("prompt", game_design, original, page_builder) != _ecosystem_key(
        "prompt",
        game_design,
        other_target,
        page_builder,
    )


def test_nonblocking_research_guard_is_live_on_complete_planner() -> None:
    assert getattr(
        complete_planner._retrieve_implementation_evidence,
        "_mmm_nonblocking_planner_research",
        False,
    )
    assert getattr(
        complete_planner.collect_ecosystem_seed_bundle,
        "_mmm_nonblocking_planner_research",
        False,
    )


def test_io_defaults_are_speed_tuned(monkeypatch) -> None:
    monkeypatch.delenv("MMM_DISCOVERY_WORKERS", raising=False)
    monkeypatch.delenv("MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS", raising=False)

    assert parallel_runtime._env_workers("MMM_DISCOVERY_WORKERS", 8, maximum=32) == 24
    assert EcosystemDiscoveryClient().timeout_seconds == 8.0
