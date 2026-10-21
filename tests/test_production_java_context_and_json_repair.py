"""Production (4)(3) regression: 38240 tokens cannot enter a 32768 slot.

The production compiler remains authoritative for Java semantics. These tests
verify that host-owned one-member context and open-source JSON repair act
before code reaches the unchanged AST/typed-state admission gates.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.construction_java_context import (
    scoped_java_member_messages,
)
from minecraft_mod_ai.construction_source_repair import assemble_java_implementation
from minecraft_mod_ai.model_adapters.base import AdapterConfig
from minecraft_mod_ai.model_router import ModelRouter


FRAME = {"kind": "method", "prefix": "public boolean valid(Object o) {\n",
         "suffix": "\n}"}


class _Router(ModelRouter):
    def __init__(self, adapter):
        self.adapter = adapter

    def _generation_adapter(self, _role):
        return self.adapter.config, self.adapter

    @contextmanager
    def _generation_scope(self, _config):
        yield


def _messages():
    # Mirrors the production pipeline: huge owner inventory in the prior
    # inspected API message, followed by the current-member work order.
    irrelevant = {"source": "UNRELATED_JDK_OWNER_" * 7000,
                  "target_api": [{"owner": "verified.Api", "name": "run"}],
                  "class_members": []}
    return [
        {"role": "user", "content": json.dumps({
            "inspected_target_api": irrelevant,
            "materialized_prerequisite_api": [],
            "assigned_behaviors": [{"behavior_id": "one", "intent": "call the API"}],
        })},
        {"role": "user", "content": json.dumps({
            "response_source_frame": FRAME,
            "requirement": "Keep this behavior",
            "implementation_requirements": ["Must return true"],
            "assigned_behavior": {"behavior_id": "one", "intent": "call the API"},
            "current_unit_purpose": "Return verified result",
            "host_signature": FRAME["prefix"],
            "declared_member_api": ["public boolean valid(Object o)"],
            "previous_complete_members": ["OLD_JAVA_BODY_" * 5000],
        })},
    ]


def test_scoping_keeps_exact_current_behavior_and_verified_apis_not_old_bodies():
    projected = scoped_java_member_messages(_messages(), FRAME)
    assert len(projected) == 2
    data = json.loads(projected[-1]["content"])
    assert data["assigned_behavior"]["behavior_id"] == "one"
    assert data["implementation_requirements"] == ["Must return true"]
    assert data["inspected_target_api"]["target_api"][0]["owner"] == "verified.Api"
    assert data["immutable_signature"]["prefix"] == FRAME["prefix"]
    assert "UNRELATED_JDK_OWNER_" not in str(projected)
    assert "OLD_JAVA_BODY_" not in str(projected)
    assert "response_source_frame" not in data


def test_exact_server_overflow_scopes_before_model_inference(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda msgs, **kwargs: tuple(msgs),
    )
    calls = {"inference": [], "accounting": []}

    def accounting(request):
        serialized = json.dumps(request.messages)
        tokens = 38240 if ("UNRELATED_JDK_OWNER_" in serialized
                           or "OLD_JAVA_BODY_" in serialized) else 800
        calls["accounting"].append(tokens)
        return SimpleNamespace(input_tokens=tokens, context_tokens=32768,
                               remaining_tokens=32768 - tokens)

    def generate(request):
        calls["inference"].append(request)
        assert "UNRELATED_JDK_OWNER_" not in str(request.messages)
        assert "OLD_JAVA_BODY_" not in str(request.messages)
        return json.dumps({"implementation": "return true;"})

    adapter = SimpleNamespace(
        config=AdapterConfig(role="coder", adapter="llama_cpp",
                             max_context=32768, max_new_tokens=8192),
        input_context_accounting=accounting,
        generate=generate,
    )
    result = _Router(adapter).generate_java_source(
        "coder", _messages(), data_name="submit_java_implementation_1",
        source_frame=FRAME,
    )
    assert result == FRAME["prefix"] + "return true;" + FRAME["suffix"]
    assert calls["accounting"] == [38240, 800]
    assert len(calls["inference"]) == 1


def test_valid_json_code_skips_json_repair(monkeypatch):
    import json_repair
    def forbidden(*args, **kwargs):
        raise AssertionError("repair must not mutate already-valid JSON")
    monkeypatch.setattr(json_repair, "loads", forbidden)
    assert assemble_java_implementation(
        json.dumps({"implementation": "return true;"}), FRAME,
    ) == FRAME["prefix"] + "return true;" + FRAME["suffix"]


def test_malformed_outer_json_is_repaired_and_java_is_ast_validated():
    # Missing closing outer brace; the decoded body itself is intact.
    result = assemble_java_implementation(
        '{"implementation": "return true;"', FRAME,
    )
    assert result == FRAME["prefix"] + "return true;" + FRAME["suffix"]


def test_json_repair_cannot_accept_invalid_java():
    with pytest.raises(ValueError):
        assemble_java_implementation('{"implementation": "if () return true;"', FRAME)


def test_json_repair_cannot_add_other_fields():
    with pytest.raises(ValueError, match="CONSTRUCTION_SOURCE_IMPLEMENTATION_INVALID"):
        assemble_java_implementation('{"something_else": "return true;"', FRAME)


def test_non_scopable_overflow_fails_before_model_runs(monkeypatch):
    from minecraft_mod_ai.model_adapters.base import ModelConfigurationError
    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda msgs, **kwargs: tuple(msgs),
    )
    adapter = SimpleNamespace(
        config=AdapterConfig(role="coder", adapter="llama_cpp",
                             max_context=32768, max_new_tokens=8192),
        input_context_accounting=lambda request: SimpleNamespace(
            input_tokens=38240, context_tokens=32768,
            remaining_tokens=-5472,
        ),
        generate=lambda request: pytest.fail("must not dispatch impossible input"),
    )
    with pytest.raises(ModelConfigurationError, match="NOT_SCOPABLE"):
        _Router(adapter).generate_java_source(
            "coder", [{"role": "user", "content": "unknown context"}],
            data_name="submit_java_implementation_1", source_frame=FRAME,
        )


def test_scoped_compiler_repair_keeps_repair_diagnostic_and_verified_api():
    response = '{"implementation":"return bad;"}'
    from minecraft_mod_ai.construction_source_repair import focused_java_correction_messages
    original = [
        {"role": "user", "content": json.dumps({
            "response_source_frame": FRAME,
            "current_unit_purpose": "Use verified helper",
            "assigned_behavior": {"behavior_id": "real", "intent": "real effect"},
            "inspected_target_api": {"target_api": [
                {"owner": "verified.RealApi", "name": "go", "kind": "METHOD"}],
                "class_members": [], "typed_input_producers": []},
        })},
    ]
    retry = focused_java_correction_messages(
        original, frame=FRAME, response=response,
        diagnostic="cannot find symbol: bad",
    )
    scoped = scoped_java_member_messages(retry, FRAME)
    payload = json.loads(scoped[-1]["content"])
    assert "cannot find symbol" in payload["diagnostic"]
    assert "Change the actual invalid Java expression" in payload["correction_obligation"]
    assert payload["api_evidence"]["selected_target_api"][0]["owner"] == "verified.RealApi"
    assert payload["previous_implementation"] == "return bad;"
