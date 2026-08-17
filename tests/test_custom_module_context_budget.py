from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.custom_module_generator import _coder_project_context_budget
from minecraft_mod_ai.scale_policy import ScalePolicy


class _Registry:
    def __init__(self, *, max_context: int, max_input_tokens: int, max_new_tokens: int) -> None:
        self.config = SimpleNamespace(
            max_context=max_context,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

    def role(self, profile: str, role: str):
        assert profile == "Qwen3.5-9B_6GB"
        assert role == "coder"
        return self.config


def _router(*, max_context: int = 32768, max_input_tokens: int = 0, max_new_tokens: int = 8192):
    return SimpleNamespace(
        profile="Qwen3.5-9B_6GB",
        registry=_Registry(
            max_context=max_context,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
    )


def test_qwen35_32k_context_uses_more_than_legacy_12k_page() -> None:
    budget = _coder_project_context_budget(
        _router(),
        ScalePolicy(),
        fast_mode=False,
    )

    assert budget == 40 * 1024
    assert budget > 12 * 1024


def test_fast_mode_keeps_four_kib_limit() -> None:
    assert _coder_project_context_budget(
        _router(),
        ScalePolicy(),
        fast_mode=True,
    ) == 4 * 1024


def test_host_context_cap_remains_authoritative() -> None:
    assert _coder_project_context_budget(
        _router(),
        ScalePolicy(model_context_bytes=16 * 1024),
        fast_mode=False,
    ) == 16 * 1024


def test_unknown_router_keeps_legacy_safe_fallback() -> None:
    assert _coder_project_context_budget(
        SimpleNamespace(),
        ScalePolicy(),
        fast_mode=False,
    ) == 12 * 1024


def test_explicit_max_input_tokens_override_full_context_derivation() -> None:
    assert _coder_project_context_budget(
        _router(max_context=131072, max_input_tokens=10000, max_new_tokens=8192),
        ScalePolicy(),
        fast_mode=False,
    ) == (10000 - 4096) * 2
