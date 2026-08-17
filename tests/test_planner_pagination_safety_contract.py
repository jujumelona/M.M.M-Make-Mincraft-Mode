from __future__ import annotations
from types import SimpleNamespace
import pytest
from minecraft_mod_ai import complete_planner as planner_module
from minecraft_mod_ai.planner_pagination_safety_contract import install
from minecraft_mod_ai.spec import SpecValidationError

def _planner():
    install(planner_module)
    return planner_module.CompleteGameDesignPlanner(SimpleNamespace())

def test_outline_repeated_cursor_fails_closed(monkeypatch) -> None:
    planner = _planner()
    first_page = {'production_batches': [{'batch_id': 'first', 'scope': 'first scope', 'depends_on_batches': [], 'deliverables': ['one'], 'exports': []}], 'complete': False, 'next_cursor': 'c1'}
    repeated = {'production_batches': [{'batch_id': 'second', 'scope': 'second scope', 'depends_on_batches': [], 'deliverables': ['two'], 'exports': []}], 'complete': False, 'next_cursor': 'c1'}
    monkeypatch.setattr(planner_module, '_generate_json_page_with_repair', lambda *args, **kwargs: repeated)
    with pytest.raises(SpecValidationError, match='did not advance'):
        planner._collect_one_request_page_outline(first_page=first_page, base_request={}, page_index=0, page_count=1)
