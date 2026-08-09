from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _RESEARCH_SHARD_CONFIG_BYTES,
    _RESEARCH_SHARD_INTEGRATION_TYPE,
    _complete_research_facts,
    _ensure_research_shards,
    _research_config_size,
    _research_sha256,
)
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner


def _base_proposal():
    return SimpleNamespace(
        spec=SimpleNamespace(contents=(), boss=None),
    )


def _large_research_design() -> dict:
    domain_count = 90
    candidate_count = 180
    domains = [
        {
            "domain_id": f"domain_{index}",
            "objective": f"Implement objective {index} exactly.",
            "requirements": [f"requirement-{index}-a", f"requirement-{index}-b"],
            "evidence_kinds": ["minecraft_api", "compatibility"],
            "queries": [f"query-{index}-a", f"query-{index}-b"],
            "providers": ["official_docs", "github"],
            "depends_on": ([f"domain_{index - 1}"] if index else []),
        }
        for index in range(domain_count)
    ]
    technical_domains = [
        {
            "domain_id": f"domain_{index}",
            "strategy": "adaptive_per_query",
            "queries": [
                {
                    "query_sha256": f"sha256:query-{index}",
                    "strategy": "corrective_multi_hop",
                    "primary": {
                        "schema_version": "mmm/retrieval-receipt-v1",
                        "query_family": "fabric_runtime",
                        "minecraft_version": "1.20.1",
                        "loader": "fabric",
                        "mappings": "yarn-1.20.1+build.1",
                        "query_hash": f"sha256:primary-{index}",
                        "corpus_snapshot_hash": "sha256:official-corpus",
                        "quality": "strong",
                        "coverage": 0.95,
                        "correction_required": True,
                        "correction_queries": [f"correction query {index}"],
                        "hits": [
                            {
                                "evidence_id": f"evidence-{index}-a",
                                "document_id": f"document-{index}-a",
                                "content_sha256": f"sha256:content-{index}-a",
                                "revision": "1.20.1",
                                "minecraft_versions": ["1.20.1"],
                                "score": 0.99,
                                "channels": ["api"],
                                "title": "UNTRUSTED_TITLE_MUST_NOT_APPEAR",
                                "excerpt": "UNTRUSTED_EXCERPT_MUST_NOT_APPEAR",
                            }
                        ],
                    },
                    "corrections": [
                        {
                            "query_hash": f"sha256:correction-{index}",
                            "corpus_snapshot_hash": "sha256:official-corpus",
                            "quality": "strong",
                            "coverage": 1.0,
                            "correction_required": False,
                            "correction_queries": [],
                            "hits": [
                                {
                                    "evidence_id": f"evidence-{index}-b",
                                    "document_id": f"document-{index}-b",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        for index in range(domain_count)
    ]
    candidates = [
        {
            "candidate_id": f"huggingface:owner/model-{index}",
            "provider": "huggingface_models",
            "resource_kind": "ai_model",
            "title": "UNTRUSTED_CANDIDATE_TITLE_MUST_NOT_APPEAR",
            "summary": "UNTRUSTED_CANDIDATE_SUMMARY_MUST_NOT_APPEAR",
            "license_id": "apache-2.0",
            "license_policy": "reviewable_model_license",
            "minecraft_version": "not_applicable",
            "loader": "not_applicable",
            "compatibility": f"unverified-{index}",
            "reuse_status": "candidate_only_metadata_not_weights",
            "evidence_sha256": f"sha256:candidate-{index}",
            "metadata": {
                "revision_sha": f"revision-{index:04d}",
                "pipeline_tag": "automatic-speech-recognition",
                "library_name": "transformers",
                "last_modified": "2026-08-09T00:00:00Z",
                "private": False,
                "gated": index % 2 == 0,
                "disabled": False,
                "card": {
                    "license_id": "apache-2.0",
                    "license_url": "https://example.invalid/license",
                    "license_evidence": "model_card_metadata",
                    "datasets": [f"owner/dataset-{index}", f"owner/extra-{index}"],
                    "languages": ["ko", f"lang-{index}"],
                },
                "format_inventory": {
                    "has_safetensors": True,
                    "has_gguf": False,
                    "has_onnx": index % 3 == 0,
                    "unsafe_serialization_files": ["weights.bin"],
                    "repository_code_files": ["modeling.py", "config.py"],
                },
                "untrusted_description": "UNTRUSTED_DESCRIPTION_MUST_NOT_APPEAR",
            },
        }
        for index in range(candidate_count)
    ]
    requirements = [
        {
            "schema_version": "mmm/technique-requirement-v1",
            "requirement_id": "npc_ai" if index == 0 else "quest_ai",
            "domain_id": "npc_dialogue" if index == 0 else "quest_generation",
            "capability_kind": "ai_inference",
            "objective": "Responsive NPC dialogue" if index == 0 else "Offline quest generation",
            "target": {"minecraft_version": "1.20.1", "loader": "fabric"},
            "allowed_topologies": ["local_sidecar"],
            "authority": {"game_state_mutation": "server_only"},
            "hardware": {"benchmark_on_declared_target": True},
            "latency": {"p95_budget_ms": 250 if index == 0 else 2500},
            "privacy": {"raw_input_sensitive": index == 0},
            "offline_required": index == 1,
            "required_gates": [f"gate-{index}-a", f"gate-{index}-b"],
            "required_tests": [f"test-{index}-a", f"test-{index}-b"],
            "deterministic_fallback": f"fallback-{index}",
            "research_queries": [f"technology-query-{index}"],
        }
        for index in range(2)
    ]
    return {
        "title": "Large researched mod",
        "_research_brief": {
            "schema_version": "mmm/central-research-brief-v1",
            "summary": "UNTRUSTED_RESEARCH_SUMMARY_MUST_NOT_APPEAR",
            "origin": "planner_classification",
            "brief_sha256": "sha256:brief",
            "domains": domains,
        },
        "_technical_evidence": {
            "schema_version": "mmm/central-evidence-graph-v1",
            "brief_sha256": "sha256:brief",
            "evidence_sha256": "sha256:evidence",
            "unresolved_official_domains": [],
            "domains": technical_domains,
        },
        "_ecosystem_discovery": {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "aggregate_schema_version": "mmm/ecosystem-seed-aggregate-v1",
            "status": "available",
            "route_sha256": "sha256:routes",
            "query_sha256": "sha256:queries",
            "route_count": 1,
            "processed_route_count": 1,
            "remaining_route_count": 0,
            "routes_complete": True,
            "candidate_count": candidate_count,
            "coverage": "seed_only",
            "pages": [
                {
                    "schema_version": "mmm/ecosystem-discovery-page-v1",
                    "research_domain_id": "voice_models",
                    "provider": "huggingface_models",
                    "query_sha256": "sha256:hf-query",
                    "minecraft_version": "1.20.1",
                    "loader": "fabric",
                    "target_profile": "speech_ai",
                    "returned": candidate_count,
                    "provider_total_estimate": candidate_count,
                    "provider_truncated": False,
                    "provider_result_limit": None,
                    "next_cursor": "",
                    "download_performed": False,
                    "authorization": "none",
                    "page_sha256": "sha256:hf-page",
                    "candidates": candidates,
                }
            ],
            "errors": [],
            "collection_receipt": {
                "schema_version": "mmm/ecosystem-route-collection-receipt-v1",
                "route_page_count": 1,
                "route_pages_sha256": "sha256:route-pages",
            },
        },
        "_technology_radar": {
            "schema_version": "mmm/technology-radar-page-v1",
            "aggregate_schema_version": "mmm/technology-radar-aggregate-v1",
            "source_sha256": "sha256:technology-source",
            "radar_sha256": "sha256:technology-radar",
            "target": {"minecraft_version": "1.20.1", "loader": "fabric", "java_version": "17"},
            "target_evidence_policy": {
                "official_exact_version_receipt_required": True,
                "receipt_schema": "mmm/official-target-evidence-v1",
            },
            "classification": {"ai_requested": True, "voice_requested": True},
            "voice_contract": {
                "activated": True,
                "speaker_identity": "tts_model",
                "expression": {
                    "representation": "utterance_local_pattern_trace",
                    "fields": ["time", "energy", "entropy", "f0", "attack", "pause"],
                },
            },
            "requirements": requirements,
            "pagination": {"complete": True, "total_requirements": 2},
            "collection_receipt": {"page_count": 1, "pages_sha256": "sha256:technology-pages"},
        },
    }


def _joined_values(facts: list[dict], source_type: str, path: str) -> list[object]:
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        if fact["source_type"] == source_type and fact["path"] == path:
            groups.setdefault(fact["source_id"], []).append(fact)
    result: list[object] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item["value_part_index"])
        if ordered[0]["value_type"] == "string":
            result.append("".join(str(item["value"]) for item in ordered))
        else:
            result.append(ordered[0]["value"])
    return result


def test_full_research_graph_is_losslessly_bound_to_bounded_generic_shards() -> None:
    design = _large_research_design()
    expected_facts, _ = _complete_research_facts(design)
    modules = _ensure_research_shards(
        (
            ProductionModule("gameplay_core", "custom_java", {}),
            ProductionModule(
                "model_research_guess",
                "integration",
                {"integration_type": _RESEARCH_SHARD_INTEGRATION_TYPE},
            ),
            ProductionModule(
                "research_consumer",
                "custom_java",
                {},
                depends_on=("model_research_guess",),
            ),
            ProductionModule(
                "voice_sidecar",
                "integration",
                {"integration_type": "mmm_local_ai_sidecar"},
            ),
        ),
        design,
        _base_proposal(),
    )
    shards = [
        module
        for module in modules
        if module.config.get("integration_type") == _RESEARCH_SHARD_INTEGRATION_TYPE
    ]
    assert len(expected_facts) > 500
    assert len(shards) > 1
    assert all(module.kind == "integration" for module in shards)
    assert all(_research_config_size(module.config) <= _RESEARCH_SHARD_CONFIG_BYTES for module in shards)
    assert [module.config["shard_index"] for module in shards] == list(range(len(shards)))
    assert all(module.config["shard_count"] == len(shards) for module in shards)
    assert shards[0].depends_on == ()
    assert all(
        shards[index].depends_on == (shards[index - 1].module_id,)
        for index in range(1, len(shards))
    )
    assert len({module.module_id for module in modules}) == len(modules)
    final_shard_id = shards[-1].module_id
    actual_modules = [module for module in modules if module not in shards]
    assert actual_modules
    assert all(
        final_shard_id in module.depends_on
        for module in actual_modules
    )
    assert all(
        module.config["artifact"]["kind"] == "project_research_ledger_json"
        and module.config["artifact"]["write_mode"] == "exact_json_resource_only"
        and module.config["artifact"]["generate_java_or_gameplay_feature"] is False
        and module.config["artifact"]["target_path"].startswith(
            ".minecraft_ai/research/"
        )
        for module in shards
    )
    assert any(module.config.get("integration_type") == "mmm_local_ai_sidecar" for module in modules)

    actual_facts = [fact for module in shards for fact in module.config["facts"]]
    assert {fact["fact_id"] for fact in actual_facts} == {
        fact["fact_id"] for fact in expected_facts
    }
    receipt = shards[0].config["receipt"]
    assert receipt["fact_count"] == len(actual_facts)
    assert receipt["facts_sha256"] == _research_sha256(expected_facts)
    assert all(
        module.required_gates == ("research ledger integrity",)
        for module in shards
    )

    assert set(_joined_values(actual_facts, "research_domain", "/domain_id")) == {
        f"domain_{index}" for index in range(90)
    }
    assert set(_joined_values(actual_facts, "technical_query", "/query_sha256")) == {
        f"sha256:query-{index}" for index in range(90)
    }
    assert set(_joined_values(actual_facts, "ecosystem_candidate", "/candidate_id")) == {
        f"huggingface:owner/model-{index}" for index in range(180)
    }
    assert "owner/dataset-179" in _joined_values(actual_facts, "ecosystem_candidate", "/datasets/0")
    assert set(_joined_values(actual_facts, "technology_requirement", "/objective")) == {
        "Responsive NPC dialogue",
        "Offline quest generation",
    }
    assert set(_joined_values(actual_facts, "technology_requirement", "/required_gates/0")) == {
        "gate-0-a",
        "gate-1-a",
    }
    assert set(_joined_values(actual_facts, "technology_requirement", "/required_tests/1")) == {
        "test-0-b",
        "test-1-b",
    }
    rendered = json.dumps([module.config for module in shards], ensure_ascii=False)
    assert "UNTRUSTED_TITLE_MUST_NOT_APPEAR" not in rendered
    assert "UNTRUSTED_EXCERPT_MUST_NOT_APPEAR" not in rendered
    assert "UNTRUSTED_CANDIDATE_TITLE_MUST_NOT_APPEAR" not in rendered
    assert "UNTRUSTED_CANDIDATE_SUMMARY_MUST_NOT_APPEAR" not in rendered
    assert "UNTRUSTED_DESCRIPTION_MUST_NOT_APPEAR" not in rendered
    assert "UNTRUSTED_RESEARCH_SUMMARY_MUST_NOT_APPEAR" not in rendered

    repeated = _ensure_research_shards((), design, _base_proposal())
    assert [module.module_id for module in repeated] == [
        module.module_id for module in shards
    ]


def test_long_safe_value_is_utf8_chunked_without_data_loss() -> None:
    objective = "한글목표-" * 900
    design = {
        "_research_brief": {
            "brief_sha256": "sha256:long",
            "domains": [
                {
                    "domain_id": "long_domain",
                    "objective": objective,
                    "requirements": [],
                    "evidence_kinds": [],
                    "queries": [],
                    "providers": [],
                    "depends_on": [],
                }
            ],
        }
    }
    modules = _ensure_research_shards((), design, _base_proposal())
    shards = [module for module in modules if module.config.get("integration_type") == _RESEARCH_SHARD_INTEGRATION_TYPE]
    facts = [fact for module in shards for fact in module.config["facts"]]

    assert _joined_values(facts, "research_domain", "/objective") == [objective]
    assert all(_research_config_size(module.config) <= _RESEARCH_SHARD_CONFIG_BYTES for module in shards)
    objective_parts = [
        fact
        for fact in facts
        if fact["source_type"] == "research_domain" and fact["path"] == "/objective"
    ]
    assert len(objective_parts) > 1
    assert all(len(str(fact["value"]).encode("utf-8")) <= 2048 for fact in objective_parts)


class _BranchRouter:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def generate_text(self, role, messages, **kwargs):
        del messages, kwargs
        assert role == "planner"
        return json.dumps(self.responses.pop(0))


def _frontdoor_base():
    return MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one shard anchor item"
    )


def _patch_branch_frontdoor(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _frontdoor_base()
    game_design = {
        "title": "Branch coverage",
        "pitch": "Bind code-owned research in every planner response shape.",
        "core_loop": ["Use the requested system"],
        "progression": [],
        "combat": {"player_verbs": [], "enemy_roles": []},
        "mod_context": {"vanilla_integration": [], "compatibility_targets": []},
        "modules": [],
        "assets": [],
        "acceptance_tests": ["The research receipt remains bound."],
        "_research_brief": {
            "schema_version": "mmm/central-research-brief-v1",
            "brief_sha256": "sha256:branch-brief",
            "origin": "planner_classification",
            "domains": [
                {
                    "domain_id": "branch_domain",
                    "objective": "Preserve this objective.",
                    "requirements": ["preserve"],
                    "evidence_kinds": ["testing"],
                    "queries": ["branch query"],
                    "providers": ["official_docs"],
                    "depends_on": [],
                }
            ],
        },
    }
    monkeypatch.setattr(
        planner_module.GameDesignPlanner,
        "plan",
        lambda self, prompt, media_paths=(): (game_design, base),
    )
    monkeypatch.setattr(
        planner_module,
        "_retrieve_implementation_evidence",
        lambda *args, **kwargs: {
            "schema_version": "mmm/central-evidence-graph-v1",
            "brief_sha256": "sha256:branch-brief",
            "evidence_sha256": "sha256:branch-evidence",
            "domains": [
                {
                    "domain_id": "branch_domain",
                    "strategy": "routed_to_other_providers",
                    "queries": [],
                }
            ],
            "unresolved_official_domains": [],
        },
    )
    monkeypatch.setattr(
        planner_module,
        "collect_ecosystem_seed_bundle",
        lambda *args, **kwargs: {
            "schema_version": "mmm/ecosystem-seed-bundle-v1",
            "route_sha256": "sha256:branch-routes",
            "pages": [],
        },
    )
    monkeypatch.setattr(
        planner_module,
        "collect_technology_radar",
        lambda *args, **kwargs: {
            "schema_version": "mmm/technology-radar-page-v1",
            "radar_sha256": "sha256:branch-radar",
            "target": {"minecraft_version": "1.20.1", "loader": "fabric"},
            "target_evidence_policy": {
                "official_exact_version_receipt_required": True,
            },
            "voice_contract": {"activated": False},
            "requirements": [],
        },
    )


def _module_payload() -> dict:
    return {
        "module_id": "branch_runtime",
        "kind": "custom_java",
        "config": {"feature": "branch coverage"},
        "depends_on": [],
        "required_gates": ["GameTest"],
    }


@pytest.mark.parametrize("response_shape", ("modules", "module_batches", "production_batches"))
def test_every_planner_response_shape_receives_code_owned_research_shards(
    monkeypatch: pytest.MonkeyPatch,
    response_shape: str,
) -> None:
    _patch_branch_frontdoor(monkeypatch)
    if response_shape == "modules":
        responses = [
            {
                "modules": [_module_payload()],
                "assets": [],
                "audio": [],
                "acceptance_tests": ["Branch runtime is observable."],
            }
        ]
    elif response_shape == "module_batches":
        responses = [
            {
                "module_batches": [
                    {
                        "batch_id": "branch_batch",
                        "scope": "Implement branch runtime.",
                        "depends_on_batches": [],
                    }
                ],
                "assets": [],
                "audio": [],
                "acceptance_tests": ["Branch runtime is observable."],
            },
            {
                "modules": [_module_payload()],
                "complete": True,
                "next_cursor": "",
            },
        ]
    else:
        responses = [
            {
                "production_batches": [
                    {
                        "batch_id": "branch_batch",
                        "scope": "Implement branch runtime.",
                        "depends_on_batches": [],
                        "deliverables": ["branch runtime"],
                        "exports": ["branch_runtime"],
                    }
                ],
                "complete": True,
                "next_cursor": "",
            },
            {
                "modules": [_module_payload()],
                "assets": [],
                "audio": [],
                "acceptance_tests": ["Branch runtime is observable."],
                "completed_deliverables": ["branch runtime"],
                "complete": True,
                "next_cursor": "",
            },
        ]

    router = _BranchRouter(responses)
    proposal = CompleteGameDesignPlanner(router).plan("Build branch coverage.")
    shards = [
        module
        for module in proposal.modules
        if module.config.get("integration_type") == _RESEARCH_SHARD_INTEGRATION_TYPE
    ]
    assert shards
    facts = [fact for module in shards for fact in module.config["facts"]]
    assert "branch_domain" in _joined_values(facts, "research_domain", "/domain_id")
    runtime = next(module for module in proposal.modules if module.module_id == "branch_runtime")
    assert shards[-1].module_id in runtime.depends_on
    assert proposal.approval_hash == proposal.calculate_hash()
    assert not router.responses
