from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import adaptive_retrieval_contract
from minecraft_mod_ai import small_model_compacting_adapter


def _base_tool_loop(self, *, adapter, request, runtime, stage, role):
    del self, adapter, request, runtime, stage, role
    return "base"


def test_adaptive_replacement_does_not_inherit_bypassed_contract_markers() -> None:
    class Router:
        pass

    base = _base_tool_loop
    base._mmm_lossless_context_compaction = True
    Router._generate_with_tools = base
    module = SimpleNamespace(ModelRouter=Router)

    adaptive_retrieval_contract._install_router_loop(module)

    live = Router._generate_with_tools
    assert getattr(live, "__mmm_progress_aware_retrieval_v1__", False) is True
    assert getattr(live, "_mmm_lossless_context_compaction", False) is False
    assert live.__wrapped__ is base


def test_compaction_install_ignores_inherited_marker_on_non_compaction_owner() -> None:
    class Router:
        pass

    base = _base_tool_loop
    base._mmm_lossless_context_compaction = True
    Router._generate_with_tools = base
    module = SimpleNamespace(ModelRouter=Router)

    small_model_compacting_adapter.install(module)

    live = Router._generate_with_tools
    assert live is not base
    assert small_model_compacting_adapter._is_live_compaction_wrapper(live) is True
    assert live.__wrapped__ is base
