"""Native member decoding rejects compilation units before AST admission."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from types import SimpleNamespace

import pytest
from lark import Lark, UnexpectedInput

from minecraft_mod_ai.java_fragment_grammar import native_java_implementation_grammar
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest


@lru_cache
def _parser(grammar):
    # Independent parser of the emitted GBNF subset, not the production Java parser.
    token = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|[a-z][a-z0-9-]*')

    def convert(match):
        value = match[0]
        if value == '""':
            return ''
        if value.startswith('"'):
            return value
        if value.startswith('['):
            return '/' + value.replace('/', r'\/') + '/'
        return value.replace('-', '_')

    lark = '\n'.join(token.sub(convert, line).replace('::=', ':') for line in grammar.splitlines())
    return Lark(lark, parser="earley", lexer="dynamic", start="root")


def _accepted(grammar, implementation):
    try:
        _parser(grammar).parse(json.dumps({"implementation": implementation}, ensure_ascii=False))
    except UnexpectedInput:
        return False
    return True


@pytest.mark.parametrize("kind,implementation", [
    ("field", "this"),
    ("field", "new java.util.HashMap<String, Integer>()"),
    ("field", "String.class"),
    ("field", '"package example; public class Example {}"'),
    ("field", "new Runnable() { public void run() { work(); } }"),
    ("field", "() -> { work(); return 7; }"),
    ("method", "int n = 1; if (n > 0) { work(n); }"),
    ("method", 'event.register((player, world) -> { send(player, "hello"); });'),
    ("method", "/* package is a comment */ packageName.call();"),
    ("method", "class Local { void work() {} } new Local().work();"),
    ("constructor", "super(7); this.value = 8;"),
    ("method", 'String value = "quote: \\\" and slash \\\\";'),
    ("method", "char quote = '\\''; char slash = '\\\\';"),
    ("method", 'String 한국어 = "문자열";'),
    ("method", 'String text = """\nhello " quoted\n""";'),
    ("field", '"""\nhello " quoted\n"""'),
    ("method", 'String text = """\nhello\\\nworld\n""";'),
    ("field", '"""\nhello\\\nworld\n"""'),
    ("field", r'"\u0041"'),
])
def test_member_grammar_preserves_java_expressions_and_nested_blocks(kind, implementation):
    assert _accepted(native_java_implementation_grammar(kind), implementation)


@pytest.mark.parametrize("kind,implementation", [
    ("field", "package example; public class Feature {}"),
    ("field", "import example.Feature; new Feature()"),
    ("field", "public final class Feature {}"),
    ("field", "class Feature {}"),
    ("field", "this; public void escaped() {}"),
    ("method", "package example; public class Feature {}"),
    ("method", "import java.util.Map; work();"),
    ("method", "public final class Feature {}"),
    ("method", "work(); } public void escaped() { work();"),
    ("method", "/* */ package escaped; /* */"),
    ("method", "/* */ public class Escaped {} /* */"),
    ("field", "1 /* */; public void escaped() {} /* */"),
    ("field", r'"\u0022; public void escaped() {} String x = \u0022"'),
])
def test_member_grammar_cannot_emit_compilation_units_or_escape_the_frame(kind, implementation):
    assert not _accepted(native_java_implementation_grammar(kind), implementation)


def test_json_unicode_cannot_hide_java_delimiters_from_the_sampler():
    payload = r'{"implementation":"\"\u0022; public void escaped() {} String x = \u0022\""}'
    with pytest.raises(UnexpectedInput):
        _parser(native_java_implementation_grammar("field")).parse(payload)


def test_member_grammar_has_no_source_or_identifier_length_ceiling():
    grammar = native_java_implementation_grammar("field")
    assert _accepted(grammar, "meaningful" * 30)
    assert _accepted(grammar, '"' + "large field literal " * 80 + '"')
    assert not _accepted(grammar, "")
    assert _accepted(native_java_implementation_grammar("field", True), "")
    assert _accepted(native_java_implementation_grammar("method"), "")


@pytest.mark.parametrize("install_wrapper", [False, True])
def test_real_source_payload_keeps_the_member_grammar_with_and_without_decode_wrappers(install_wrapper):
    from minecraft_mod_ai import llama_server_hardware_policy as hardware
    from minecraft_mod_ai.llama_structured_decode_policy import bind_structured_decode_policy

    if install_wrapper:
        bind_structured_decode_policy(hardware)
    frame = {"kind": "field", "prefix": "public final Feature Callback = ", "suffix": ";"}
    request = GenerationRequest(response_format="json", response_schema=None, metadata={
        "mmm_source_implementation": True, "mmm_source_declaration_frame": frame,
        "mmm_strict_source_data": True, "mmm_force_non_thinking": True})
    adapter = SimpleNamespace(config=AdapterConfig(role="coder", adapter="llama_cpp",
        model_id="MiMo", max_context=32768, max_new_tokens=8192, extra={"runtime_contract": "mimo"}))
    payload = hardware._server_payload(adapter, request)
    assert "grammar" in payload
    assert "json_schema" not in payload and "response_format" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert _accepted(payload["grammar"], "this")
    assert not _accepted(payload["grammar"], "package example; public final class Feature {}")


def test_production_native_sampler_excludes_wrappers_and_still_emits_executable_java(tmp_path, monkeypatch):
    from test_direct_feature_source import SourceCoder, _fixture, _run_java
    from test_first_pass_java_data_transport import DataRouter
    from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
    from minecraft_mod_ai.model_adapters import llama_cpp_adapter
    from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter
    from minecraft_mod_ai import llama_exact_context

    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context",
                        lambda messages, **kwargs: messages)
    monkeypatch.setattr(llama_exact_context, "capacity_safe_payload", lambda url, payload, **kwargs: dict(payload))
    bad = 'package example; import java.util.Set; public final class Feature { public Feature self; }'
    payloads = []

    class SamplerCoder(SourceCoder):
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            adapter = LlamaCppAdapter(AdapterConfig(role="coder", adapter="llama_cpp", model_id="MiMo",
                max_context=32768, max_new_tokens=8192, extra={"runtime_contract": "mimo"}))
            monkeypatch.setattr(adapter, "_server_url", lambda request: "http://local/v1")
            monkeypatch.setattr(adapter, "input_context_accounting", lambda request: SimpleNamespace(
                input_tokens=1000, context_tokens=32768, remaining_tokens=31768))

            def completion(url, payload):
                payloads.append(payload)
                assert not _accepted(payload["grammar"], bad)
                task = next(m for m in reversed(payload["messages"]) if m["role"] == "user")
                source = SourceCoder.generate_java_source(
                    self, role, [task], data_name=data_name, source_frame=source_frame)
                body = source[len(source_frame["prefix"]):-len(source_frame["suffix"])]
                assert _accepted(payload["grammar"], body)
                return {"content": json.dumps({"implementation": body})}

            monkeypatch.setattr(llama_cpp_adapter, "_completion_message", completion)
            return DataRouter(adapter).generate_java_source(
                role, messages, data_name=data_name, source_frame=source_frame)

    behavior, spec, context = _fixture()
    source = compose_first_pass_java(SamplerCoder(), spec, [], construction_context=context, behaviors=[behavior])
    _run_java(tmp_path, source)
    assert len(payloads) == 4
    assert all("json_schema" not in payload for payload in payloads)
