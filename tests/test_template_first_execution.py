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


def test_source_grounded_requirement_resolves_real_event_before_java(monkeypatch):
    spec, behavior, prompt, context, opts = _ready()
    behavior.pop("event_binding")
    behavior.update(source_status="grounded", source_quote="On known interaction")
    monkeypatch.setattr(
        "minecraft_mod_ai.source_owned_hook.choose_source_hook",
        lambda router, row, callbacks: callbacks[0],
    )
    received = []
    def render(*args, **kwargs):
        received.extend(kwargs["behaviors"])
        return "compiled checked rule"
    monkeypatch.setattr(route, "render_checked_rule", render)
    monkeypatch.setattr(host, "_verification_boundary", lambda: lambda *a, **kw: None)
    assert _compose_preferred_source(
        object(), spec, context, prompt, [behavior], **opts,
    ) == "compiled checked rule"
    assert received[0]["event_binding"] == "inspected:event"


def test_unsupported_occurrence_does_not_use_fake_template(monkeypatch):
    from minecraft_mod_ai.source_occurrence_error import SourceHookUnavailable
    spec, behavior, prompt, context, _ = _ready()
    behavior.pop("event_binding")
    behavior.update(source_status="grounded", source_quote="On known interaction")
    def no_hook(router, row, callbacks):
        raise SourceHookUnavailable("reward", reason="no_real_hook", inspected=len(callbacks))
    monkeypatch.setattr("minecraft_mod_ai.source_owned_hook.choose_source_hook", no_hook)
    assert route.resolve_inspected_template_behaviors(
        object(), spec, [behavior], callbacks=prompt["available_fabric_callbacks"],
    ) is None


def test_optional_callback_binding_survives_checkpoint_roundtrip():
    from minecraft_mod_ai.construction_task import BehaviorRequirement
    record = BehaviorRequirement(
        behavior_id="b", requirement_id="r", intent="Place block",
        source_quote="Place block", trigger="Player places block",
        effects="Update the block", event_binding="checked.Event#PLACED",
    )
    assert BehaviorRequirement(**record.to_dict()).event_binding == "checked.Event#PLACED"
    legacy = BehaviorRequirement(
        behavior_id="b", requirement_id="r", intent="Place block",
        source_quote="Place block", trigger="Player places block",
        effects="Update the block",
    )
    assert "event_binding" not in legacy.to_dict()
