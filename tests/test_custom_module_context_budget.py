from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.custom_module_generator import _coder_project_context_budget
from minecraft_mod_ai.model_context_budget import request_message_budget
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


def _expected_live_source_budget(router, policy: ScalePolicy | None = None) -> int:
    policy = policy or ScalePolicy()
    config = router.registry.role(router.profile, "coder")
    live_request_bytes = request_message_budget(config, ())
    return min(
        max(1024, int(policy.model_context_bytes)),
        max(1024, live_request_bytes // 2),
    )


def test_qwen35_live_32k_context_uses_more_than_legacy_12k_page() -> None:
    router = _router()
    budget = _coder_project_context_budget(
        router,
        ScalePolicy(),
        fast_mode=False,
    )

    assert budget == _expected_live_source_budget(router)
    assert budget > 12 * 1024


def test_fast_mode_keeps_four_kib_limit() -> None:
    assert _coder_project_context_budget(
        _router(),
        ScalePolicy(),
        fast_mode=True,
    ) == 4 * 1024


def test_host_context_cap_remains_authoritative() -> None:
    policy = ScalePolicy(model_context_bytes=16 * 1024)
    router = _router()
    assert _coder_project_context_budget(
        router,
        policy,
        fast_mode=False,
    ) == 16 * 1024


def test_unknown_router_keeps_safe_fallback() -> None:
    assert _coder_project_context_budget(
        SimpleNamespace(),
        ScalePolicy(),
        fast_mode=False,
    ) == 12 * 1024


def test_explicit_max_input_tokens_reduce_source_grounding_budget() -> None:
    unrestricted = _router()
    capped = _router(max_input_tokens=10000)

    capped_budget = _coder_project_context_budget(
        capped,
        ScalePolicy(),
        fast_mode=False,
    )

    assert capped_budget == _expected_live_source_budget(capped)
    assert capped_budget < _expected_live_source_budget(unrestricted)
