from __future__ import annotations
from types import SimpleNamespace
from minecraft_mod_ai.atomic_requirement_contract import compile_ir, validate_ir
from minecraft_mod_ai.visual_acceptance_scope_contract import _visual_refs

def test_asset_only_visual_atom_does_not_require_mineflayer_runtime() -> None:
    proposal = _proposal('Create a red glowing texture.', assets=(SimpleNamespace(asset_id='glow', kind='texture', prompt='red glowing texture', target_path='assets/example/textures/item/glow.png', width=16, height=16),), acceptance_tests=('The red glowing texture is visibly correct.',))
    ir = compile_ir(proposal)
    assert ir['unresolved_atom_ids'] == []
    assert ir['atoms'][0]['evidence_dimensions'] == ['visual_3d']
    proposal.game_design['_atomic_requirement_ir'] = ir
    assert validate_ir(proposal) is ir

def test_gameplay_module_atom_routes_to_runtime() -> None:
    proposal = _proposal('The frost sword freezes enemies.', modules=(SimpleNamespace(module_id='frost_sword', kind='weapon', config={'effect': 'freeze enemies'}, depends_on=(), required_gates=()),), acceptance_tests=('The frost sword freezes enemies in play.',))
    ir = compile_ir(proposal)
    assert ir['unresolved_atom_ids'] == []
    assert 'runtime' in ir['atoms'][0]['evidence_dimensions']

def test_performance_gameplay_atom_requires_runtime_and_performance() -> None:
    proposal = _proposal('The combat system must meet the declared performance latency budget.', modules=(SimpleNamespace(module_id='combat', kind='custom_java', config={'performance': 'latency budget'}, depends_on=(), required_gates=()),), acceptance_tests=('Combat meets the declared performance latency budget.',))
    ir = compile_ir(proposal)
    assert ir['unresolved_atom_ids'] == []
    assert ir['atoms'][0]['evidence_dimensions'] == ['runtime', 'performance']

def test_visual_scope_selects_only_visual_atom_and_quality_acceptance() -> None:
    proposal = SimpleNamespace(acceptance_tests=('Runtime combat works.', 'The red texture is visibly correct.', '[visual_3d] Visual output passes review.'), game_design={'_atomic_requirement_ir': {'atoms': [{'evidence_dimensions': ['runtime'], 'acceptance_refs': ['acceptance:00000000']}, {'evidence_dimensions': ['visual_3d'], 'acceptance_refs': ['acceptance:00000001']}]}})
    assert _visual_refs(proposal) == ('acceptance:00000001', 'acceptance:00000002')
