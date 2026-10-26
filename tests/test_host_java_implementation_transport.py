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
    assert request.response_schema is None  # No hidden 256-char native schema limit.
    assert not request.tools
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


def test_production_literal_newline_failure_is_returned_as_member_correction():
    # The exact output shape from production: a full declaration with literal
    # backslash-n characters and an extra escaped newline after its closing brace.
    rejected = FRAME["prefix"].replace("\n", "\\n") + (
        "    super(type, level);\\n}\\n")

    class CorrectingRouter(BackendRouter):
        def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return rejected
            return '{"implementation": "    super(type, level);"}'

    router = CorrectingRouter(None)
    assert _run(router) == FRAME["prefix"] + "    super(type, level);" + FRAME["suffix"]
    assert len(router.requests) == 2
    correction = json.loads(router.requests[1].messages[-1]["content"])
    assert correction["previous_response"] == rejected
    assert "CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID" in correction["diagnostic"]


def test_persistent_invalid_response_stops_after_one_member_correction():
    router = BackendRouter('{"implementation": "super(type, level); } void escape() {"}')
    with pytest.raises(JavaRegionParseError):
        _run(router)
    assert len(router.requests) == 2


def test_native_payload_enforces_member_boundaries_without_a_full_java_peg():
    from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
    from minecraft_mod_ai.java_fragment_grammar import native_java_implementation_grammar

    router = BackendRouter('{"implementation": "super(type, level);"}')
    _run(router)
    adapter = SimpleNamespace(config=SimpleNamespace(
        role="coder", model_id="local", max_new_tokens=2048, extra={}))
    payload = _server_payload(adapter, router.requests[0])
    assert "response_format" not in payload
    assert "json_schema" not in payload  # No schema-derived source string cap.
    assert payload["grammar"] == native_java_implementation_grammar("constructor")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_source_router_receives_exact_native_response_before_generic_json_cleanup():
    from minecraft_mod_ai.llama_structured_decode_policy import _bind_structured_generation_retry

    class NativeAdapter:
        def __init__(self):
            self.outputs = ['not JSON', '{"implementation": "super(type, level);"}']

        def generate(self, _request):
            return self.outputs.pop(0)

    _bind_structured_generation_retry(SimpleNamespace(LlamaCppAdapter=NativeAdapter))
    adapter = NativeAdapter()

    class NativeRouter(BackendRouter):
        def generate(self, request):
            self.requests.append(request)
            return adapter.generate(request)

    router = NativeRouter(None)
    assert "super(type, level);" in _run(router)
    assert json.loads(router.requests[-1].messages[-1]["content"])["previous_response"] == 'not JSON'


def test_host_removes_only_proven_surplus_outer_brace():
    # Replay from production-logs (14): the model repeated a host-owned
    # trailing brace after two legitimate registration calls.
    frame = {"kind": "method", "prefix": "public void onInitialize() {\n",
             "suffix": "\n}"}
    body = ("registerBehavior48b495a6ef6dd67d6375();\n"
            "registerBehavior0a2ee99147396b090c42();\n}")
    accepted = _run(BackendRouter(json.dumps({"implementation": body})), frame)
    assert accepted.count("registerBehavior48b495a6ef6dd67d6375();") == 1
    assert accepted.count("registerBehavior0a2ee99147396b090c42();") == 1
    assert accepted == frame["prefix"] + body[:-1].rstrip() + frame["suffix"]


def test_host_does_not_strip_valid_java_control_flow_braces():
    frame = {"kind": "method", "prefix": "public void action() {\n",
             "suffix": "\n}"}
    body = "if (true) { System.out.println(1); }"
    assert _run(BackendRouter(json.dumps({"implementation": body})), frame) == (
        frame["prefix"] + body + frame["suffix"])


def test_host_rejects_extra_declarations_even_with_surplus_brace():
    frame = {"kind": "method", "prefix": "public void action() {\n",
             "suffix": "\n}"}
    body = "return; } void inject() {"
    with pytest.raises(JavaRegionParseError):
        _run(BackendRouter(json.dumps({"implementation": body})), frame)
