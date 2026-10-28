"""Integration checks for the host-rule-first route and failure integrity."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.construction_host_source import _compose_preferred_source
from minecraft_mod_ai import construction_host_source as host
from minecraft_mod_ai import template_first_route as route


def _ready():
    spec = {"bindings": {"request_feature_entrypoint": "demo.Feature"}}
    behavior = {
        "behavior_id": "reward",
        "event_binding": "inspected:event",
        "trigger": "On known interaction",
        "effects": "Award the actual reward",
    }
    prompt = {
        "available_fabric_callbacks": [{"event_id": "inspected:event"}],
        "retained_typed_behavior": {},
        "owned_runtime_records": [],
        "materialized_prerequisite_api": [],
        "host_state_work_order_scope": {},
        "package": "demo", "public_class": "Feature",
    }
    context = {"host_direct_feature_source": True}
    opts = {
        "source_context": (), "instructions": "Original requirements",
        "state_access": [], "mutation_receipts": [], "incremental_state": {},
    }
    return spec, behavior, prompt, context, opts


def test_checked_rule_is_compiled_before_publishing(monkeypatch):
    spec, behavior, prompt, context, opts = _ready()
    calls = []
    monkeypatch.setattr(route, "render_checked_rule", lambda *args, **kwargs: "compiled java")
    def verifier(*args, **kwargs):
        calls.append((args[0], kwargs["package"]))
    monkeypatch.setattr(host, "_verification_boundary", lambda: verifier)
    source = _compose_preferred_source(
        object(), spec, context, prompt, [behavior], **opts)
    assert source == "compiled java"
    assert calls == [("compiled java", "demo")]


def test_checked_rule_failure_never_degrades_into_unrelated_java(monkeypatch):
    spec, behavior, prompt, context, opts = _ready()
    monkeypatch.setattr(
        route, "render_checked_rule", lambda *a, **kw: "uncompilable")
    def failed_verifier(*args, **kwargs):
        raise ValueError("original effect cannot be compiled")
    monkeypatch.setattr(host, "_verification_boundary", lambda: failed_verifier)
    with pytest.raises(ValueError, match="original effect"):
        _compose_preferred_source(object(), spec, context, prompt, [behavior], **opts)
