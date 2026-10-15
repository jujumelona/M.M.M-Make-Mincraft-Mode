"""Regressions for the production trace's memory-index 16/24 cycle."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.construction_declaration_choices import memory_schemas
from minecraft_mod_ai.construction_memory import ConstructionMemory
from minecraft_mod_ai.construction_state import ConstructionState


def _memory(count=32):
    memory = ConstructionMemory()
    rows = [{'symbol_id': f'example.Type{index}', 'kind': 'CLASS'} for index in range(count)]
    result = {'matches': rows}
    memory.remember('symbols', 'example', 0, result)
    return memory, result


def test_memory_index_cannot_repeat_production_16_24_cycle():
    memory, _ = _memory()
    assert memory.index_offset == 24
    memory.select_index(16)
    assert 24 not in memory.index_choices()
    with pytest.raises(ValueError, match='CONSTRUCTION_MEMORY_INDEX_INVALID'):
        memory.select_index(24)


def test_every_memory_page_remains_reachable_without_revisiting_one():
    memory, _ = _memory(80)
    seen = {memory.index_offset}
    while choices := memory.index_choices():
        memory.select_index(choices[0])
        assert memory.index_offset not in seen
        seen.add(memory.index_offset)
    assert seen == set(range(0, 80, 8))
    assert len(memory.records) == 80


def test_navigation_deduplication_survives_checkpoint_restore():
    memory, result = _memory()
    memory.select_index(16)
    saved = memory.export()
    restored = ConstructionMemory()
    restored.restore(saved, lambda *_: result)
    assert restored.index_choices() == memory.index_choices()
    assert 24 not in restored.index_choices()
    assert restored.payload() == memory.payload()


def test_recall_cannot_cycle_evicted_facts_in_same_decision():
    memory, _ = _memory()
    keys = list(memory.records)
    for key in keys[:16]:
        memory.recall(key)
    assert keys[0] not in memory.active
    with pytest.raises(ValueError, match='CONSTRUCTION_MEMORY_RECALL_INVALID'):
        memory.recall(keys[0])
    state = ConstructionState({}, 'example.Owner', 'Owner.java')
    state.memory = memory
    while memory.index_offset:
        memory.select_index(memory.index_choices()[0])
    recall_ids = [key for branch in memory_schemas(state)
                  for key in branch['properties'].get('evidence_id', {}).get('enum', [])]
    assert not set(keys[:8]) & set(recall_ids)


def test_new_decision_can_recall_old_exact_facts_again():
    state = ConstructionState({}, 'example.Owner', 'Owner.java')
    state.enter('member_parameters')
    memory, _ = _memory()
    state.memory = memory
    memory.select_index(16)
    state.advance()
    assert 24 in memory.index_choices()
    assert len(memory.records) == 32


def test_growing_last_catalogue_page_exposes_new_evidence():
    memory, _ = _memory(27)
    memory.select_index(16)
    assert 24 not in memory.index_choices()
    memory.remember('symbols', 'more', 0, {'matches': [{'symbol_id': 'example.Added'}]})
    assert memory.index_offset == 24
    memory.select_index(8)
    assert 16 not in memory.index_choices()
    assert len(memory.records) == 28
