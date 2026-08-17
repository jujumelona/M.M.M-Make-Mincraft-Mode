from __future__ import annotations
import json
from types import SimpleNamespace
import pytest
import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _RESEARCH_SHARD_CONFIG_BYTES, _RESEARCH_SHARD_INTEGRATION_TYPE, _complete_research_facts, _ensure_research_shards, _research_config_size, _research_sha256
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner

def _base_proposal():
    return SimpleNamespace(spec=SimpleNamespace(contents=(), boss=None))

def _joined_values(facts: list[dict], source_type: str, path: str) -> list[object]:
    groups: dict[str, list[dict]] = {}
    for fact in facts:
        if fact['source_type'] == source_type and fact['path'] == path:
            groups.setdefault(fact['source_id'], []).append(fact)
    result: list[object] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item['value_part_index'])
        if ordered[0]['value_type'] == 'string':
            result.append(''.join((str(item['value']) for item in ordered)))
        else:
            result.append(ordered[0]['value'])
    return result

def test_long_safe_value_is_utf8_chunked_without_data_loss() -> None:
    objective = '한글목표-' * 900
    design = {'_research_brief': {'brief_sha256': 'sha256:long', 'domains': [{'domain_id': 'long_domain', 'objective': objective, 'requirements': [], 'evidence_kinds': [], 'queries': [], 'providers': [], 'depends_on': []}]}}
    modules = _ensure_research_shards((), design, _base_proposal())
    shards = [module for module in modules if module.config.get('integration_type') == _RESEARCH_SHARD_INTEGRATION_TYPE]
    facts = [fact for module in shards for fact in module.config['facts']]
    assert _joined_values(facts, 'research_domain', '/objective') == [objective]
    assert all((_research_config_size(module.config) <= _RESEARCH_SHARD_CONFIG_BYTES for module in shards))
    objective_parts = [fact for fact in facts if fact['source_type'] == 'research_domain' and fact['path'] == '/objective']
    assert len(objective_parts) > 1
    assert all((len(str(fact['value']).encode('utf-8')) <= 2048 for fact in objective_parts))

class _BranchRouter:

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def generate_text(self, role, messages, **kwargs):
        del messages, kwargs
        assert role == 'planner'
        return json.dumps(self.responses.pop(0))

def _frontdoor_base():
    return MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one shard anchor item')

def _module_payload() -> dict:
    return {'module_id': 'branch_runtime', 'kind': 'custom_java', 'config': {'feature': 'branch coverage'}, 'depends_on': [], 'required_gates': ['GameTest']}
