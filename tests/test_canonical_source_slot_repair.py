"""Regressions for source-slot placeholders in content artifact-graph generation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from jsonschema.exceptions import ValidationError

from minecraft_mod_ai import canonical_generators
from minecraft_mod_ai import fixed_template_generation
from minecraft_mod_ai import implementation_template_renderer


_PACKAGE_AND_CLASS = r"(?s)\\bpackage\\s+p\\s*;.*\\bclass\\s+Good\\b"
_VALID_JAVA = "package p; public class Good { public static int value() { return 42; } }"


def _generate(monkeypatch, candidates, *, source=True):
    seen = []
    candidates = iter(candidates)
    slot_schema = {
        "type": "string",
        "minLength": 1,
        "maxLength": 16384,
        "pattern": _PACKAGE_AND_CLASS,
        "x-mmm-contract-profile": "source_repair",
        "x-mmm-string-class": "source",
    } if source else {"type": "string", "minLength": 5}
    manifest = {
        "task": "Generate Minecraft implementation",
        "rules": [],
        "inputs": [{"name": "identity", "type": "identity"}, {"name": "spec", "type": "specification"}],
        "outputs": [{"name": "artifact", "type": "code_fragment"}, {"name": "receipt", "type": "receipt"}],
    }
    spec = {
        "leaf_id": "test/registry",
        "context_id": "ctx",
        "language": "java",
        "requirement": "Create functional registry logic",
        "bindings": {"package_name": "p", "class_name": "Good"},
        "slots": [{"name": "code", "description": "Complete Java registry source", "schema": slot_schema}],
        "render_mold": "{{code}}",
        "output_schema": {"type": "string", "pattern": _PACKAGE_AND_CLASS} if source else
                         {"type": "string", "minLength": 5},
    }
    monkeypatch.setattr(canonical_generators, "load_template", lambda _leaf: manifest)
    monkeypatch.setattr(
        canonical_generators,
        "deterministic_model_map",
        lambda _router, items, worker, **_kwargs: [worker(item) for item in items],
    )
    monkeypatch.setattr(implementation_template_renderer, "render_template",
                        lambda _mold, values: values["code"])

    def model(_router, _role, messages, **kwargs):
        seen.append((list(messages), kwargs["response_schema"]))
        return next(candidates)

    monkeypatch.setattr(fixed_template_generation, "generate_fixed_template_value", model)
    authority = SimpleNamespace(
        types=SimpleNamespace(validate_input=lambda *_: None, validate_output=lambda *_: None)
    )
    return (
        lambda: canonical_generators.generate_canonical_leaf(
            {"identity": "registry", "spec": spec},
            leaf_id="test/registry",
            router=object(),
            authority=authority,
        ),
        seen,
    )


def test_placeholder_is_repaired_before_canonical_source_publication(monkeypatch):
    run, seen = _generate(monkeypatch, ["placeholder", _VALID_JAVA])
    result = run()

    assert result["artifact"] == _VALID_JAVA
    assert result["receipt"]["leaf_id"] == "test/registry"
    assert len(seen) == 2
    assert "pattern" not in seen[0][1]  # llama transport cannot enforce the host regex
    assert _PACKAGE_AND_CLASS in seen[0][0][0]["content"]  # exact host constraint is given to coder
    assert "REJECTED" in seen[1][0][-1]["content"]  # focused model repair feedback
    assert "placeholder" in seen[1][0][-1]["content"]


def test_persistent_placeholder_never_publishes_an_artifact(monkeypatch):
    run, seen = _generate(monkeypatch, ["placeholder", "placeholder", _VALID_JAVA])
    with pytest.raises(ValidationError, match="does not match"):
        run()
    assert len(seen) == 2  # fixed, bounded repair; no third call or silent fallback


def test_non_source_slot_does_not_retry_invalid_scalar(monkeypatch):
    run, seen = _generate(monkeypatch, ["bad", "complete"], source=False)
    with pytest.raises(ValidationError, match="too short"):
        run()
    assert len(seen) == 1


def test_valid_source_does_not_consume_repair_attempt(monkeypatch):
    run, seen = _generate(monkeypatch, [_VALID_JAVA, "placeholder"])
    assert run()["artifact"] == _VALID_JAVA
    assert len(seen) == 1
