"""Java source uses ordinary decoding and the host's exact declaration frame."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import model_router as router_module
from minecraft_mod_ai.model_adapters.base import ModelBackendError


class FakeRouter:
    def __init__(self, source="public void foo() { return; }", failure=None):
        self.source, self.failure, self.requests = source, failure, []

    def _generation_adapter(self, _role):
        return SimpleNamespace(), SimpleNamespace(generate=self._native)

    @contextmanager
    def _generation_scope(self, _config):
        yield

    def _native(self, request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.source


@pytest.fixture(autouse=True)
def _isolate_context_fit(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, **_kwargs: messages,
    )


def _execute(router, frame=None):
    return router_module.ModelRouter.generate_java_source(
        router, "coder", [], data_name="construction_unit_foo",
        source_frame=frame or {
            "kind": "method", "prefix": "public void foo() {", "suffix": "}",
        },
    )


def test_java_source_starts_without_recursive_grammar_or_recovery_cap():
    fake = FakeRouter()
    assert _execute(fake) == fake.source
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.response_format == "text"
    assert "mmm_java_source_frame" not in request.metadata
    assert "mmm_native_java_recovery" not in request.metadata
    assert "mmm_output_token_ceiling" not in request.metadata
    prompt = "\n".join(message["content"] for message in request.messages)
    assert "public void foo() {" in prompt


def test_complete_long_declaration_is_not_truncated_by_a_recovery_ceiling():
    source = "public void foo() {" + "\n".join(
        f"int local{i} = {i};" for i in range(1500)
    ) + "\n}"
    fake = FakeRouter(source=source)
    assert _execute(fake) == source
    assert len(fake.requests) == 1


def test_source_cannot_bypass_host_owned_declaration_signature():
    with pytest.raises(router_module.ModelConfigurationError,
                       match="CONSTRUCTION_SOURCE_SIGNATURE_INVALID"):
        _execute(FakeRouter(source="public void other() { return; }"))


def test_uninitialized_field_remains_a_valid_frame_alternative():
    frame = {"kind": "field", "prefix": "private int count = ",
             "suffix": ";", "empty_value": "private int count;"}
    assert _execute(FakeRouter(source="private int count;"), frame) == "private int count;"


@pytest.mark.parametrize("extra", [
    " static { System.setProperty(\"escaped\", \"true\"); }",
    " public void other() {}",
    " } class Extra { static {}",
])
def test_raw_declaration_cannot_add_sibling_members_or_initializers(extra):
    from minecraft_mod_ai.java_region_parser import JavaRegionParseError

    with pytest.raises(JavaRegionParseError):
        _execute(FakeRouter(source="public void foo() { return; }" + extra))


def test_backend_failure_does_not_replay_a_second_generation_strategy():
    fake = FakeRouter(failure=ModelBackendError(
        role="coder", model_id="mimo", cause=RuntimeError("connection denied"),
    ))
    with pytest.raises(ModelBackendError):
        _execute(fake)
    assert len(fake.requests) == 1


def test_legacy_recovery_metadata_cannot_shrink_dynamic_context(monkeypatch):
    from minecraft_mod_ai import generation_output_budget as budget

    monkeypatch.setattr(budget, "apply_payload_generation_budget",
                        lambda _payload, **_kwargs: {"max_tokens": 30000})
    monkeypatch.setattr(budget, "dynamic_output_budget_enabled", lambda _config: True)
    request = SimpleNamespace(response_format="text", metadata={
        "mmm_output_token_ceiling": 4096, "mmm_native_java_recovery": True,
        "mmm_strict_source_data": True,
    })
    assert budget.apply_request_generation_budget(
        {}, config=SimpleNamespace(), request=request,
    )["max_tokens"] == 30000
