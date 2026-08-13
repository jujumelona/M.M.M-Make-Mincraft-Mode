from __future__ import annotations

import minecraft_mod_ai.complete_planner as complete_planner
import minecraft_mod_ai.parallel_runtime_contract as parallel_runtime
from minecraft_mod_ai.ecosystem_discovery import EcosystemDiscoveryClient
from minecraft_mod_ai.planning_stall_guard_contract import _planning_seed_brief


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


def test_seed_projection_keeps_coverage_without_query_explosion(monkeypatch) -> None:
    monkeypatch.delenv("MMM_ECOSYSTEM_SEED_QUERIES_PER_DOMAIN", raising=False)
    original = _brief()
    projected = _planning_seed_brief(original)

    assert [item["domain_id"] for item in projected["domains"]] == ["gameplay", "visuals"]
    assert projected["domains"][0]["providers"] == original["domains"][0]["providers"]
    assert projected["domains"][1]["providers"] == original["domains"][1]["providers"]
    assert projected["domains"][0]["queries"] == ["q1"]
    assert projected["domains"][1]["queries"] == ["v1"]
    assert original["domains"][0]["queries"] == ["q1", "q2", "q3"]


def test_stall_guard_is_live_on_complete_planner() -> None:
    assert getattr(complete_planner._retrieve_implementation_evidence, "_mmm_stall_guard", False)
    assert getattr(complete_planner.collect_ecosystem_seed_bundle, "_mmm_stall_guard", False)


def test_io_defaults_are_speed_tuned(monkeypatch) -> None:
    monkeypatch.delenv("MMM_DISCOVERY_WORKERS", raising=False)
    monkeypatch.delenv("MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS", raising=False)

    assert parallel_runtime._env_workers("MMM_DISCOVERY_WORKERS", 8, maximum=32) == 24
    assert EcosystemDiscoveryClient().timeout_seconds == 8.0
