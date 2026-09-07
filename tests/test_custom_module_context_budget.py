from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.custom_module_generator import _coder_project_context_budget
from minecraft_mod_ai.scale_policy import ScalePolicy


class _Registry:
    def __init__(self, *, max_context: int, max_input_tokens: int, max_new_tokens: int) -> None:
        self.config = SimpleNamespace(
            role="coder",
            adapter="llama_cpp",
            provider="local",
            model_id="test/qwen",
            max_context=max_context,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            extra={
                "runtime_contract": "qwen",
                "decode_hotpath": "t4_mtp",
                "runtime_context_default": 32768,
            },
        )

    def role(self, profile: str, role: str):
        assert profile == "Qwen3.5-9B_6GB"
        assert role == "coder"
        return self.config


def _router(
    *,
    max_context: int = 262144,
    max_input_tokens: int = 0,
    max_new_tokens: int = 8192,
):
    return SimpleNamespace(
        profile="Qwen3.5-9B_6GB",
        registry=_Registry(
            max_context=max_context,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        ),
    )


def test_large_live_context_does_not_expand_initial_atomic_source_page() -> None:
    assert _coder_project_context_budget(
        _router(),
        ScalePolicy(),
        fast_mode=False,
    ) == 4 * 1024


def test_fast_mode_keeps_the_same_small_initial_page_limit() -> None:
    assert _coder_project_context_budget(
        _router(),
        ScalePolicy(),
        fast_mode=True,
    ) == 4 * 1024


def test_host_context_cap_cannot_expand_atomic_initial_page() -> None:
    policy = ScalePolicy(model_context_bytes=16 * 1024)
    assert _coder_project_context_budget(
        _router(),
        policy,
        fast_mode=False,
    ) == 4 * 1024


def test_unknown_router_keeps_small_safe_limit() -> None:
    assert _coder_project_context_budget(
        SimpleNamespace(),
        ScalePolicy(),
        fast_mode=False,
    ) == 4 * 1024


def test_smaller_live_input_capacity_can_reduce_below_atomic_cap() -> None:
    budget = _coder_project_context_budget(
        _router(max_input_tokens=1024),
        ScalePolicy(),
        fast_mode=False,
    )
    assert 1024 <= budget <= 4 * 1024
