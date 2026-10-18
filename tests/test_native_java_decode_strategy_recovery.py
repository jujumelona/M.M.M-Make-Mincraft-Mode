"""Regression for genuine native llama GBNF decode loops and bounded recovery."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import model_router as router_module
from minecraft_mod_ai.llama_completion_liveness_contract import (
    LlamaCompletionDeadlineExceeded, LlamaDegenerateOutputError,
)
from minecraft_mod_ai.model_adapters.base import ModelBackendError


class FakeRouter:
    def __init__(self, failure, source="public void foo() { return; }"):
        self.failure = failure
        self.source = source
        self.requests = []

    def _generation_adapter(self, _role):
        return (SimpleNamespace(), SimpleNamespace(generate=self._native))

    @contextmanager
    def _generation_scope(self, _config):
        yield

    def _native(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise self.failure
        return self.source


@pytest.fixture(autouse=True)
def _isolate_context_fit(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, **_kwargs: messages,
    )


def _execute(router):
    return router_module.ModelRouter.generate_java_source(
        router, "coder", [], data_name="construction_unit_foo",
        source_frame={"kind": "method", "prefix": "public void foo() {", "suffix": "}"},
    )


@pytest.mark.parametrize("failure", [
    LlamaDegenerateOutputError("periodic grammar"),
    LlamaCompletionDeadlineExceeded("absolute decode bound"),
    ModelBackendError(
        role="coder", model_id="mimo",
        cause=LlamaDegenerateOutputError("LLAMA_DEGENERATE_REPEATED_OUTPUT"),
    ),
])
def test_native_java_loop_switches_once_to_bounded_ungrammared_raw_java(failure):
    fake = FakeRouter(failure)
    assert _execute(fake) == fake.source
    assert len(fake.requests) == 2
    first, second = fake.requests
    assert first.metadata["mmm_java_source_frame"]["kind"] == "method"
    assert second.metadata["mmm_native_java_recovery"] is True
    assert "mmm_java_source_frame" not in second.metadata
    assert second.metadata["mmm_output_token_ceiling"] == 4096
    assert second.response_format == "text"


def test_unrelated_server_exception_does_not_retry():
    fake = FakeRouter(ModelBackendError(
        role="coder", model_id="mimo", cause=RuntimeError("connection denied"),
    ))
    with pytest.raises(ModelBackendError):
        _execute(fake)
    assert len(fake.requests) == 1


def test_recovery_cannot_bypass_signed_declaration_signature():
    fake = FakeRouter(
        LlamaDegenerateOutputError("grammar repetition"),
        source="public void other() { return; }",
    )
    with pytest.raises(router_module.ModelConfigurationError,
                       match="CONSTRUCTION_BOUNDED_SOURCE_SIGNATURE_INVALID"):
        _execute(fake)
    assert len(fake.requests) == 2


def test_only_explicit_native_java_recovery_overrides_dynamic_output_ceiling(monkeypatch):
    from minecraft_mod_ai import generation_output_budget as budget

    monkeypatch.setattr(
        budget, "apply_payload_generation_budget",
        lambda _payload, **_kwargs: {"max_tokens": 30000},
    )
    monkeypatch.setattr(budget, "dynamic_output_budget_enabled", lambda _config: True)
    config = SimpleNamespace()
    base = {"mmm_output_token_ceiling": 4096}
    request = SimpleNamespace(response_format="text", metadata=base)
    assert budget.apply_request_generation_budget({}, config=config, request=request)[
        "max_tokens"
    ] == 30000

    request.metadata = {
        **base, "mmm_native_java_recovery": True, "mmm_strict_source_data": True,
    }
    assert budget.apply_request_generation_budget({}, config=config, request=request)[
        "max_tokens"
    ] == 4096
