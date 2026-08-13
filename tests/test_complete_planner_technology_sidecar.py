from __future__ import annotations

import json

import pytest

import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _SIDECAR_EXECUTION_CAPABILITIES,
    _ensure_technology_sidecar,
    _implementation_prompt,
)
from minecraft_mod_ai.complete_spec import CompleteProposal, ProductionModule
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.spec import SpecValidationError


def _base_proposal():
    return MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create one technology anchor item"
    )


def _radar(*capabilities: str) -> dict[str, object]:
    return {
        "schema_version": "mmm/technology-radar-page-v1",
        "requirements": [
            {
                "capability_kind": capability,
                "required_gates": ["exact_compatibility", "model_license"],
                "required_tests": ["fabric_runtime", "latency_p95"],
                "deterministic_fallback": f"Disable {capability} safely.",
            }
            for capability in capabilities
        ],
    }


def _sidecars(modules: tuple[ProductionModule, ...]) -> list[ProductionModule]:
    return [
        module
        for module in modules
        if module.kind == "integration"
        and module.config.get("integration_type") == "mmm_local_ai_sidecar"
    ]


def test_radar_capabilities_create_one_strict_sidecar_and_derived_gates() -> None:
    modules = _ensure_technology_sidecar(
        (),
        _radar(
            "voice_transport",
            "speech_synthesis",
            "ai_inference",
            "language_intersection",
            "speech_synthesis",
        ),
        _base_proposal(),
    )

    sidecars = _sidecars(modules)
    assert len(sidecars) == 1
    sidecar = sidecars[0]
    assert sidecar.config == {
        "integration_type": "mmm_local_ai_sidecar",
        "port": 8765,
        "timeout_ms": 5000,
        "max_request_bytes": 262144,
        "max_response_bytes": 262144,
        "max_in_flight": 4,
        "capabilities": ["ai_inference", "speech_synthesis"],
        "authentication": "external_token",
    }
    assert "technology:ai_inference:gate:exact_compatibility" in sidecar.required_gates
    assert "technology:speech_synthesis:test:latency_p95" in sidecar.required_gates
    assert (
        "technology:ai_inference:fallback:Disable ai_inference safely."
        in sidecar.required_gates
    )
    assert not any("voice_transport" in gate for gate in sidecar.required_gates)


def test_prompt_bounds_duplicate_capability_details_but_sidecar_keeps_all_gates() -> None:
    radar = {
        "schema_version": "mmm/technology-radar-aggregate-v1",
        "radar_sha256": "sha256:" + "a" * 64,
        "target": {"minecraft_version": "1.20.1", "loader": "fabric"},
        "target_evidence_policy": {
            "official_exact_version_receipt_required": True,
            "receipt_schema": "mmm/official-target-evidence-v1",
        },
        "classification": {"ai_requested": True},
        "voice_contract": {"activated": False},
        "collection_receipt": {
            "schema_version": "mmm/technology-page-collection-receipt-v1",
            "page_count": 2,
            "pages_sha256": "sha256:" + "b" * 64,
        },
        "requirements": [
            {
                "requirement_id": "npc_ai_inference",
                "domain_id": "npc_dialogue",
                "capability_kind": "ai_inference",
                "allowed_topologies": ["local_sidecar"],
                "authority": {"game_state_mutation": "server_only"},
                "hardware": {"benchmark_on_declared_target": True},
                "latency": {"real_time_required": True},
                "privacy": {"raw_input_sensitive": False},
                "offline_required": False,
                "required_gates": ["npc_gate"],
                "required_tests": ["npc_test"],
                "deterministic_fallback": "Use scripted NPC dialogue.",
            },
            {
                "requirement_id": "quest_ai_inference",
                "domain_id": "quest_generation",
                "capability_kind": "ai_inference",
                "allowed_topologies": ["local_sidecar"],
                "authority": {"game_state_mutation": "server_only"},
                "hardware": {"benchmark_on_declared_target": True},
                "latency": {"real_time_required": False},
                "privacy": {"raw_input_sensitive": False},
                "offline_required": True,
                "required_gates": ["quest_gate"],
                "required_tests": ["quest_test"],
                "deterministic_fallback": "Use authored quest templates.",
            },
        ],
    }

    rendered = _implementation_prompt(
        "Add NPC dialogue and generated quests.",
        {"title": "Two AI systems", "_technology_radar": radar},
    )
    encoded_context = rendered.split(
        "Compact authoritative planning context:\n", 1
    )[1].split("\n\nCreate only the paginated production outline.", 1)[0]
    technology = json.loads(encoded_context)["research_outline"][
        "technology_radar"
    ]

    assert technology["capability_counts"] == {"ai_inference": 2}
    assert technology["requirement_count"] == 2
    assert technology["representative_requirement_count"] == 1
    assert technology["requirement_view_complete"] is False
    assert technology["collection_receipt"]["page_count"] == 2
    assert technology["requirements"][0]["requirement_id"] == "npc_ai_inference"
    assert "quest_ai_inference" not in rendered
    assert len(radar["requirements"]) == 2

    sidecar = _sidecars(
        _ensure_technology_sidecar((), radar, _base_proposal())
    )[0]
    assert "technology:ai_inference:gate:npc_gate" in sidecar.required_gates
    assert "technology:ai_inference:gate:quest_gate" in sidecar.required_gates
    assert "technology:ai_inference:test:quest_test" in sidecar.required_gates
    assert (
        "technology:ai_inference:fallback:Use authored quest templates."
        in sidecar.required_gates
    )


def test_existing_duplicate_sidecars_collapse_and_dependencies_are_remapped() -> None:
    modules = (
        ProductionModule(
            "z_voice_sidecar",
            "integration",
            {
                "integration_type": "mmm_local_ai_sidecar",
                "capabilities": ["voice_conversion"],
            },
            depends_on=("voice_assets",),
        ),
        ProductionModule("voice_assets", "custom_java", {}),
        ProductionModule(
            "a_voice_sidecar",
            "integration",
            {
                "integration_type": "mmm_local_ai_sidecar",
                "capabilities": ["agent_tool_use"],
            },
        ),
        ProductionModule(
            "npc_dialogue",
            "custom_java",
            {},
            depends_on=("z_voice_sidecar",),
        ),
    )

    normalized = _ensure_technology_sidecar(
        modules,
        _radar("speech_recognition", "translation"),
        _base_proposal(),
    )

    sidecars = _sidecars(normalized)
    assert len(sidecars) == 1
    assert sidecars[0].module_id == "a_voice_sidecar"
    assert sidecars[0].config["capabilities"] == [
        "speech_recognition",
        "translation",
    ]
    assert sidecars[0].depends_on == ("voice_assets",)
    consumer = next(
        module for module in normalized if module.module_id == "npc_dialogue"
    )
    assert consumer.depends_on == ("a_voice_sidecar",)


def test_sidecar_id_search_has_no_small_collision_attempt_cap() -> None:
    occupied = tuple(
        ProductionModule(
            (
                "mmm_local_ai_sidecar"
                if index == 1
                else f"mmm_local_ai_sidecar_{index}"
            ),
            "integration",
            {"integration_type": "unrelated_adapter"},
        )
        for index in range(1, 1501)
    )

    normalized = _ensure_technology_sidecar(
        occupied,
        _radar("agent_tool_use"),
        _base_proposal(),
    )

    sidecars = _sidecars(normalized)
    assert len(sidecars) == 1
    assert sidecars[0].module_id == "mmm_local_ai_sidecar_1501"
    assert len(normalized) == 1501


def test_no_execution_capability_removes_model_hallucinated_sidecar() -> None:
    modules = (
        ProductionModule(
            "invented_sidecar",
            "integration",
            {
                "integration_type": "mmm_local_ai_sidecar",
                "capabilities": ["speech_synthesis"],
            },
        ),
        ProductionModule(
            "tomato_crop",
            "crop",
            {},
            depends_on=("invented_sidecar",),
        ),
    )

    normalized = _ensure_technology_sidecar(
        modules,
        _radar("voice_transport", "language_intersection"),
        _base_proposal(),
    )

    assert _sidecars(normalized) == []
    assert normalized[0].module_id == "tomato_crop"
    assert normalized[0].depends_on == ()


class _ImplementationRouter:
    def __init__(self, modules: list[dict[str, object]]) -> None:
        self.modules = modules

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        return json.dumps(
            {
                "modules": self.modules,
                "assets": [],
                "audio": [],
                "acceptance_tests": ["The requested feature has an observable fallback."],
            }
        )


def _empty_discovery_page() -> dict[str, object]:
    return {
        "schema_version": "mmm/ecosystem-seed-bundle-v1",
        "status": "empty",
        "query_sha256": "sha256:test-query",
        "route_sha256": "sha256:test-routes",
        "route_count": 0,
        "route_offset": 0,
        "processed_route_count": 0,
        "remaining_route_count": 0,
        "next_route_cursor": "",
        "routes_complete": True,
        "candidate_count": 0,
        "pages": [],
        "errors": [],
        "coverage": "test-empty",
        "authorization": "none",
        "download_performed": False,
    }


def _patch_frontdoor(monkeypatch: pytest.MonkeyPatch, prompt: str) -> None:
    base = _base_proposal()
    game_design = {
        "title": "Request-derived integration",
        "pitch": prompt,
        "mod_context": {
            "vanilla_integration": [],
            "compatibility_targets": [],
        },
        "_research_brief": {
            "schema_version": "minecraft-mod-ai/research-brief-v1",
            "domains": [
                {
                    "domain_id": "requested_feature",
                    "objective": prompt,
                    "requirements": [prompt],
                    "evidence_kinds": [],
                    "queries": [prompt],
                    "providers": [],
                    "depends_on": [],
                }
            ],
        },
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
        lambda *args, **kwargs: _empty_discovery_page(),
    )


def test_plan_binds_radar_sidecar_into_approval_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = (
        "Add AI inference, speech recognition, speech synthesis and translation "
        "to an NPC, with a deterministic text fallback."
    )
    _patch_frontdoor(monkeypatch, prompt)
    proposal = CompleteGameDesignPlanner(_ImplementationRouter([])).plan(prompt)

    sidecars = _sidecars(proposal.modules)
    assert len(sidecars) == 1
    radar_capabilities = sorted(
        {
            str(item["capability_kind"])
            for item in proposal.game_design["_technology_radar"]["requirements"]
            if item["capability_kind"] in _SIDECAR_EXECUTION_CAPABILITIES
        }
    )
    assert sidecars[0].config["capabilities"] == radar_capabilities
    assert proposal.approval_hash == proposal.calculate_hash()

    tampered = json.loads(json.dumps(proposal.to_dict()))
    tampered["modules"][0]["config"]["port"] = 18765
    with pytest.raises(SpecValidationError, match="approval_hash"):
        CompleteProposal.from_dict(tampered)


def test_farming_and_procedural_audio_plan_never_keeps_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = "Create a tomato farm and a procedural bell audio asset."
    _patch_frontdoor(monkeypatch, prompt)
    hallucinated = {
        "module_id": "hallucinated_ai_bridge",
        "kind": "integration",
        "config": {
            "integration_type": "mmm_local_ai_sidecar",
            "capabilities": ["speech_synthesis"],
        },
        "depends_on": [],
        "required_gates": [],
    }
    proposal = CompleteGameDesignPlanner(
        _ImplementationRouter([hallucinated])
    ).plan(prompt)

    assert _sidecars(proposal.modules) == []
    assert proposal.game_design["_technology_radar"]["requirements"] == []
