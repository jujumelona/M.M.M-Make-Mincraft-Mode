"""Only implementation data crosses the model boundary; the host emits Java frames."""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
from minecraft_mod_ai.java_region_parser import JavaRegionParseError


FRAME = {"kind": "constructor", "prefix": (
    "public AlienCombatHostileEntity(net.minecraft.world.entity.EntityType type, "
    "net.minecraft.world.level.Level level) {\n"
), "suffix": "\n}"}


class BackendRouter:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def _generation_adapter(self, _role):
        return SimpleNamespace(), SimpleNamespace(generate=self.generate)

    @contextmanager
    def _generation_scope(self, _config):
        yield

    def generate(self, request):
        self.requests.append(request)
        return self.response


@pytest.fixture(autouse=True)
def _context_fit(monkeypatch):
    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context",
                        lambda messages, **kwargs: messages)


def _run(router, frame=FRAME):
    return ModelRouter.generate_java_source(router, "coder", [],
        data_name="submit_java_implementation_1", source_frame=frame)


def test_constructor_body_is_json_decoded_then_host_framed():
    body = "    super(type, level);"
    router = BackendRouter(json.dumps({"implementation": body}))
    assert _run(router) == FRAME["prefix"] + body + FRAME["suffix"]
    request = router.requests[0]
    assert request.response_format == "json"
    assert request.response_schema is None  # No generic planner string-length limit.
    assert request.metadata["mmm_force_non_thinking"] is True
    assert "mmm_output_token_ceiling" not in request.metadata
    assert not request.tools


def test_json_transport_distinguishes_layout_newlines_from_java_string_escapes():
    frame = {"kind": "method", "prefix": "public String text() {\n", "suffix": "\n}"}
    body = 'String path = "C:\\\\new";\nreturn path + "\\n";'
    source = _run(BackendRouter(json.dumps({"implementation": body})), frame)
    assert source == frame["prefix"] + body + frame["suffix"]
    assert '"\\n"' in source
    assert '"C:\\\\new"' in source


def test_long_implementation_keeps_every_statement_without_transport_page_cap():
    frame = {"kind": "method", "prefix": "public void work() {\n", "suffix": "\n}"}
    body = "\n".join(f"int local{i} = {i};" for i in range(1500))
    assert len(body) > 16384
    assert _run(BackendRouter(json.dumps({"implementation": body})), frame) == (
        frame["prefix"] + body + frame["suffix"])


@pytest.mark.parametrize("response", [
    '{"wrong": "super(type, level);"}',
    '{"implementation": 1}',
    '{"implementation": "super(type, level);", "signature": "Other"}',
    '"super(type, level);"',
    'super(type, level);',
])
def test_invalid_implementation_envelope_preserves_raw_failure(response):
    with pytest.raises(ModelConfigurationError, match="CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID") as failure:
        _run(BackendRouter(response))
    assert failure.value.failed_response == response


@pytest.mark.parametrize("body", [
    'super(type, level); } public void extra() {',
    'super(type, level); } } class Extra { void extra() {',
    'public AlienCombatHostileEntity() {}',
])
def test_implementation_cannot_replace_signature_or_escape_member(body):
    with pytest.raises(JavaRegionParseError):
        _run(BackendRouter(json.dumps({"implementation": body})))


def test_empty_initializer_only_uses_host_approved_uninitialized_field():
    frame = {"kind": "field", "prefix": "private int count = ", "suffix": ";",
             "empty_value": "private int count;"}
    assert _run(BackendRouter('{"implementation": ""}'), frame) == frame["empty_value"]
    del frame["empty_value"]
    with pytest.raises(JavaRegionParseError):
        _run(BackendRouter('{"implementation": ""}'), frame)
