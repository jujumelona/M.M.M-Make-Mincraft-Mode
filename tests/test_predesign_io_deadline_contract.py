from __future__ import annotations

import inspect

from minecraft_mod_ai import pre_design_grounded_rag


def test_predesign_external_mod_transports_are_removed() -> None:
    source = inspect.getsource(pre_design_grounded_rag)
    assert "def _search_curseforge(" not in source
    assert "def _search_modrinth(" not in source


def test_transport_keeps_finite_network_timeout() -> None:
    source = inspect.getsource(pre_design_grounded_rag)
    assert "_TIMEOUT = 8.0" in source
    assert "urlopen(request, timeout=_TIMEOUT)" in source


def test_catalog_query_scheduler_uses_shared_deadline_executor() -> None:
    from minecraft_mod_ai import catalog_first_grounded_rag

    source = inspect.getsource(catalog_first_grounded_rag)
    assert 'backend._query_worker_count(len(specs))' in source
    assert 'stage="predesign-query-bundles"' in source
    assert 'iter_completed_with_deadlines' in source
