from __future__ import annotations

import inspect

from minecraft_mod_ai import custom_generation_search_contract


def test_custom_generation_fanouts_use_deadline_executor() -> None:
    source = inspect.getsource(custom_generation_search_contract)

    assert "ThreadPoolExecutor" not in source
    assert ".result()" not in source
    assert "pool.map(" not in source
    assert "iter_completed_with_deadlines" in source


def test_custom_generation_fanouts_have_named_deadline_stages() -> None:
    source = inspect.getsource(custom_generation_search_contract)

    assert "stage='custom-generation-candidates'" in source
    assert "stage='custom-generation-verification'" in source
