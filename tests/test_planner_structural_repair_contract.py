from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.evidence_first_planning import SemanticRequirementIR, _semantic_requirement_fields, build_request_catalog, compile_evidence_first_plan
from minecraft_mod_ai.reuse_planner import decompose_capability_graph


def _router_for(caps: tuple[str, ...]):
    payload = '{"intent":"semantic request","gameplay_capability_candidates":[' + ','.join(f'"{cap}"' for cap in caps) + '],"unresolved":false}'
    return SimpleNamespace(generate_text=lambda *_args, **_kwargs: payload)


def test_direct_semantic_fields_preserve_explicit_capability_without_provenance() -> None:
    ir = SemanticRequirementIR(0, 11, 'sha256:' + '0' * 64, 'trade items', ('economy.trade',), 0.95, False)
    fields = _semantic_requirement_fields('custom.explicit_design_id', ir, 'req_explicit')
    assert fields['capability'] == 'custom.explicit_design_id'


def test_real_router_multi_root_clause_explodes_to_independent_requirements() -> None:
    prompt = 'Implement the requested progression systems.'
    caps = ('mob.spawning','boss.entity','item.equipment','progression.level','item.upgrade')
    catalog = build_request_catalog(prompt, {}, router=_router_for(caps))
    got = {item['capability'] for item in catalog['requirements']}
    assert set(caps) <= got
    assert all(len(item['provides']) == 1 for item in catalog['requirements'])


def test_real_router_multi_root_plan_has_gap_and_task_chain_for_each_root() -> None:
    prompt = 'Implement the requested progression systems.'
    caps = ('mob.spawning','boss.entity','item.equipment','progression.level','item.upgrade')
    plan = compile_evidence_first_plan(prompt, {}, target_decision={'target':{'minecraft_version':'1.21.1','loader':'neoforge','source_api_family':'neoforge'}}, semantic_router=_router_for(caps))
    assert set(caps) <= {gap['capability'] for gap in plan['gap_catalog']}
    task_refs = {ref for task in plan['tasks'] for ref in task['requirement_refs']}
    assert task_refs == {req['requirement_id'] for req in plan['request_catalog']['requirements']}


def test_prompt_unknown_is_one_opaque_but_design_scope_does_not_inflate() -> None:
    graph = decompose_capability_graph('Add seasonal rune banking.')
    opaque = [node for node in graph.nodes if node.startswith('provisional:')]
    assert len(opaque) == 1
    design = {'capabilities': [f'system.feature_{index}' for index in range(96)]}
    scoped = decompose_capability_graph('Implement the declared systems.', design=design)
    assert len(scoped.nodes) == 96
    assert not any(node.startswith('provisional:') for node in scoped.nodes)
