"""Native Java decode loops must switch strategy without skipping host validation."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import model_router as router_module
from minecraft_mod_ai.llama_completion_liveness_contract import (
    LlamaCompletionDeadlineExceeded, LlamaDegenerateOutputError,
)


class FakeRouter:
    def __init__(self, failure, returned_source="public void foo() { return; }"):
        self.failure = failure
        self.returned_source = returned_source
        self.native_calls = 0
        self.json_calls = 0
        self.schema = None
        self.name = None

    def _generation_adapter(self, _role):
        return (SimpleNamespace(), SimpleNamespace(generate=self._native))

    @contextmanager
    def _generation_scope(self, _config):
        yield

    def _native(self, _request):
        self.native_calls += 1
        raise self.failure

    def generate_structured_data(self, _role, _messages, *, data_name, parameters):
        self.json_calls += 1
        self.name = data_name
        self.schema = parameters
        return {"source": self.returned_source}


@pytest.mark.parametrize("failure", [
    LlamaDegenerateOutputError("periodic grammar"),
    LlamaCompletionDeadlineExceeded("absolute decode bound"),
])
def test_native_java_loop_switches_once_to_bounded_json(monkeypatch, failure):
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, **_kwargs: messages,
    )
    fake = FakeRouter(failure)
    result = router_module.ModelRouter.generate_java_source(
        fake, "coder", [], data_name="construction_unit_foo",
        source_frame={"kind": "method", "prefix": "public void foo() {",
                      "suffix": "}"},
    )
    assert result == fake.returned_source
    assert fake.native_calls == fake.json_calls == 1
    assert fake.schema["properties"]["source"]["maxLength"] == 2400
    assert fake.name.endswith(".bounded_json_fallback")


def test_unrelated_server_exception_does_not_retry():
    fake = FakeRouter(ValueError("bad external state"))
    with pytest.raises(ValueError, match="bad external state"):
        router_module.ModelRouter.generate_java_source(
            fake, "coder", [], data_name="x",
            source_frame={"kind": "method", "prefix": "public", "suffix": "}"},
        )
    assert fake.native_calls == 1
    assert fake.json_calls == 0


def test_recovery_cannot_bypass_signed_declaration_signature(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, **_kwargs: messages,
    )
    fake = FakeRouter(
        LlamaDegenerateOutputError("grammar repetition"),
        returned_source="public void other() { return; }",
    )
    with pytest.raises(router_module.ModelConfigurationError,
                       match="CONSTRUCTION_BOUNDED_SOURCE_SIGNATURE_INVALID"):
        router_module.ModelRouter.generate_java_source(
            fake, "coder", [], data_name="construction_unit_foo",
            source_frame={"kind": "method", "prefix": "public void foo() {",
                          "suffix": "}"},
        )
    assert fake.native_calls == fake.json_calls == 1
