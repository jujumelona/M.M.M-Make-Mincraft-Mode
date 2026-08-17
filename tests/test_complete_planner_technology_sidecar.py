from __future__ import annotations
import json
import pytest
import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _SIDECAR_EXECUTION_CAPABILITIES, _ensure_technology_sidecar, _implementation_prompt
from minecraft_mod_ai.complete_spec import CompleteProposal, ProductionModule
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.spec import SpecValidationError

def _base_proposal():
    return MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one technology anchor item')

def _radar(*capabilities: str) -> dict[str, object]:
    return {'schema_version': 'mmm/technology-radar-page-v1', 'requirements': [{'capability_kind': capability, 'required_gates': ['exact_compatibility', 'model_license'], 'required_tests': ['fabric_runtime', 'latency_p95'], 'deterministic_fallback': f'Disable {capability} safely.'} for capability in capabilities]}

def _sidecars(modules: tuple[ProductionModule, ...]) -> list[ProductionModule]:
    return [module for module in modules if module.kind == 'integration' and module.config.get('integration_type') == 'mmm_local_ai_sidecar']

def test_sidecar_id_search_has_no_small_collision_attempt_cap() -> None:
    occupied = tuple((ProductionModule('mmm_local_ai_sidecar' if index == 1 else f'mmm_local_ai_sidecar_{index}', 'integration', {'integration_type': 'unrelated_adapter'}) for index in range(1, 1501)))
    normalized = _ensure_technology_sidecar(occupied, _radar('agent_tool_use'), _base_proposal())
    sidecars = _sidecars(normalized)
    assert len(sidecars) == 1
    assert sidecars[0].module_id == 'mmm_local_ai_sidecar_1501'
    assert len(normalized) == 1501

def _empty_discovery_page() -> dict[str, object]:
    return {'schema_version': 'mmm/ecosystem-seed-bundle-v1', 'status': 'empty', 'query_sha256': 'sha256:test-query', 'route_sha256': 'sha256:test-routes', 'route_count': 0, 'route_offset': 0, 'processed_route_count': 0, 'remaining_route_count': 0, 'next_route_cursor': '', 'routes_complete': True, 'candidate_count': 0, 'pages': [], 'errors': [], 'coverage': 'test-empty', 'authorization': 'none', 'download_performed': False}

def _patch_frontdoor(monkeypatch: pytest.MonkeyPatch, prompt: str) -> None:
    base = _base_proposal()
    game_design = {'title': 'Request-derived integration', 'pitch': prompt, 'mod_context': {'vanilla_integration': [], 'compatibility_targets': []}, '_research_brief': {'schema_version': 'minecraft-mod-ai/research-brief-v1', 'domains': [{'domain_id': 'requested_feature', 'objective': prompt, 'requirements': [prompt], 'evidence_kinds': ['minecraft_api'], 'queries': [prompt], 'providers': ['official_docs'], 'depends_on': []}]}}
    monkeypatch.setattr(planner_module.GameDesignPlanner, 'plan', lambda self, value, media_paths=(): (game_design, base))
    monkeypatch.setattr(planner_module, '_retrieve_implementation_evidence', lambda *args, **kwargs: {'schema_version': 'test/evidence-v1'})
    monkeypatch.setattr(planner_module, 'discover_seed_bundle', lambda *args, **kwargs: _empty_discovery_page())
