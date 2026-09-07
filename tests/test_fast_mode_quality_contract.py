from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.custom_module_generator import _coder_project_context_budget


def test_fast_mode_preserves_coder_project_context_budget() -> None:
    router = SimpleNamespace(profile="", registry=None)
    policy = SimpleNamespace(model_context_bytes=64 * 1024)

    normal_budget = _coder_project_context_budget(
        router,
        policy,
        fast_mode=False,
    )
    fast_budget = _coder_project_context_budget(
        router,
        policy,
        fast_mode=True,
    )

    assert normal_budget == 12 * 1024
    assert fast_budget == normal_budget


def test_fast_mode_still_honors_host_context_cap() -> None:
    router = SimpleNamespace(profile="", registry=None)
    policy = SimpleNamespace(model_context_bytes=8 * 1024)

    assert _coder_project_context_budget(router, policy, fast_mode=False) == 8 * 1024
    assert _coder_project_context_budget(router, policy, fast_mode=True) == 8 * 1024
