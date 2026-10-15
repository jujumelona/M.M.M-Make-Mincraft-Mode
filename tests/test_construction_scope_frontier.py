"""Growing lookup history must not become a growing model-visible frontier."""
from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.construction_declaration_choices import symbol_lookup_schemas
from minecraft_mod_ai.construction_retrieval import ConstructionEvidenceIndex
from minecraft_mod_ai.construction_state import PAGE_SIZE, ConstructionState


def _state():
    symbols = {}
    for index in range(96):
        owner = f'api.branch{index:03d}.Type'
        symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': 'Type', 'access': 1}
    owner = 'example.NewOwner'
    symbols[owner] = {'kind': 'CLASS', 'owner': owner, 'name': 'NewOwner',
                      'access': 1, 'provenance': 'construction_plan'}
    index = ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path='src/main/java/example/NewOwner.java', owner=owner, authoring_contract={})
    state = ConstructionState(symbols, owner, 'src/main/java/example/NewOwner.java', evidence_index=index)
    state.enter('operation_graph')
    state.prepare_symbols('perform an operation', kinds=('CLASS',))
    root = index.root_goal(state.symbol_goal)
    api = next(row for row in index.scope_page(root, 0)['scopes'] if row['name'] == 'api')
    goal = api['goal_id']
    offset = 0
    all_candidates = set()
    while True:
        page = index.scope_page(goal, offset)
        state.consume_lookup('scopes', goal, offset, page)
        all_candidates.update(row['goal_id'] for row in page['scopes'])
        if page['next_offset'] is None:
            break
        offset = page['next_offset']
    return state, all_candidates


def test_frontier_stays_one_page_after_many_scope_reads():
    state, _ = _state()
    assert len(state.scope_candidates()) <= PAGE_SIZE
    payload = state.payload(['scopes'])
    assert 'scopes' not in payload['lookup']
    assert len(payload['scope_candidates']) <= PAGE_SIZE
    offered = {key for branch in symbol_lookup_schemas(state)
               for key in branch['properties'].get('query', {}).get('enum', [])}
    assert offered == {row['goal_id'] for row in payload['scope_candidates']}


def test_paging_frontier_preserves_every_discovered_branch():
    state, expected = _state()
    inspected = set()
    while candidates := state.scope_candidates():
        assert len(candidates) <= PAGE_SIZE
        selected = candidates[0]['goal_id']
        assert selected not in inspected
        inspected.add(selected)
        state.consume_lookup('scopes', selected, 0, state.evidence_index.scope_page(selected, 0))
    assert expected <= inspected


def test_frontier_choices_match_after_state_checkpoint_restore():
    state, _ = _state()
    saved = state.export()
    restored, _ = _state()
    restored.restore(saved, resolve_lookup=lambda kind, query, offset:
        restored.evidence_index.scope_page(query, offset) if kind == 'scopes'
        else restored.evidence_index.page(query, offset))
    assert restored.scope_candidates() == state.scope_candidates()
