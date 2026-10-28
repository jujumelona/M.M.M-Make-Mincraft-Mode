"""Regression tests for the production value_kind/lookup infinite loop."""
from __future__ import annotations

import pytest

from minecraft_mod_ai import incremental_java_values as values
from minecraft_mod_ai.incremental_java_values_model import JavaValue


class _Builder:
    def __init__(self):
        self.value_stack = [{"slot": "argument_0", "expected_type": "int"}]
        self.lookups = 0

    def lookup(self):
        self.lookups += 1


def test_operand_lookup_cannot_cycle_forever(monkeypatch):
    builder = _Builder()
    seen = []

    def respond(_builder, schema, suffix):
        assert suffix == "value_kind"
        choices = schema["properties"]["kind"]["enum"]
        seen.append(tuple(choices))
        return {"kind": "lookup" if "lookup" in choices else "literal"}

    monkeypatch.setattr(values, "request", respond)
    author = values.ValueAuthor(builder)
    monkeypatch.setattr(author, "_render", lambda kind, expected, writable: JavaValue("7", expected))
    result = author._value_choice("int", ["lookup", "literal"], writable=False)

    assert result.source == "7"
    assert builder.lookups == values.MAX_VALUE_EVIDENCE_LOOKUPS
    assert len(seen) == values.MAX_VALUE_EVIDENCE_LOOKUPS + 1
    assert "lookup" not in seen[-1]


def test_no_grounded_value_fails_instead_of_inventing_a_literal(monkeypatch):
    builder = _Builder()

    def respond(_builder, schema, suffix):
        options = schema["properties"]["kind"]["enum"]
        return {"kind": "lookup" if "lookup" in options else "new_helper"}

    monkeypatch.setattr(values, "request", respond)
    author = values.ValueAuthor(builder)
    monkeypatch.setattr(author, "_render", lambda kind, expected, writable: None)

    with pytest.raises(RuntimeError, match="JAVA_VALUE_CONSTRUCTION_UNSUPPORTED"):
        author._value_choice("example.Unknown", ["lookup", "new_helper"], writable=False)
    assert builder.lookups == values.MAX_VALUE_EVIDENCE_LOOKUPS


def test_successful_first_lookup_remains_available(monkeypatch):
    builder = _Builder()
    calls = iter(("lookup", "local"))

    def respond(_builder, schema, suffix):
        answer = next(calls)
        assert answer in schema["properties"]["kind"]["enum"]
        return {"kind": answer}

    monkeypatch.setattr(values, "request", respond)
    author = values.ValueAuthor(builder)
    monkeypatch.setattr(author, "_render", lambda kind, expected, writable: JavaValue("player", expected))
    result = author._value_choice("example.Player", ["lookup", "local"], writable=False)
    assert result.source == "player"
    assert builder.lookups == 1
