"""Model-free proof gate for the host-rendered gameplay-rule route."""
from __future__ import annotations

from minecraft_mod_ai.template_first_route import checked_rule_template_ready


def _inputs():
    return (
        {"bindings": {"request_feature_entrypoint": "example.Feature"}},
        [{"behavior_id": "mine", "event_binding": "fabric:block_broken",
          "trigger": "Player breaks a block", "effects": "Grant the actual reward"}],
        [{"event_id": "fabric:block_broken", "parameters": ["example.Player"]}],
    )


def test_exact_inspected_event_uses_host_rendered_rule():
    spec, behaviors, callbacks = _inputs()
    assert checked_rule_template_ready(spec, behaviors, callbacks=callbacks)


def test_implicit_or_uninspected_event_never_becomes_a_guessed_callback():
    spec, behaviors, callbacks = _inputs()
    assert not checked_rule_template_ready(spec, behaviors, callbacks=[])
    behaviors[0].pop("event_binding")
    assert not checked_rule_template_ready(spec, behaviors, callbacks=callbacks)


def test_native_state_mutations_require_a_source_owner():
    spec, behaviors, callbacks = _inputs()
    for key in ("runtime_state_reads", "runtime_state_writes", "state_reads", "state_writes"):
        candidate = [{**behaviors[0], key: ["planet"]}]
        assert not checked_rule_template_ready(spec, candidate, callbacks=callbacks)


def test_retained_typed_or_ui_owners_are_not_registered_twice():
    spec, behaviors, callbacks = _inputs()
    for key in ("functions", "event_bindings", "ui_owned_records", "source_ui_actions"):
        assert not checked_rule_template_ready(
            spec, behaviors, callbacks=callbacks, retained={key: [{"id": "existing"}]})
    assert not checked_rule_template_ready(
        spec, behaviors, callbacks=callbacks,
        source_ui_callbacks=[{"action_id": "existing"}])
    assert not checked_rule_template_ready(
        spec, behaviors, callbacks=callbacks, owned_runtime_records=[{"record": 1}])


def test_broad_request_without_behaviors_is_not_fake_success():
    spec, _, callbacks = _inputs()
    assert not checked_rule_template_ready(spec, [], callbacks=callbacks)
    assert not checked_rule_template_ready({"bindings": {}}, [], callbacks=callbacks)


def test_every_behavior_must_be_explicitly_supported():
    spec, behaviors, callbacks = _inputs()
    candidate = [*behaviors, {"behavior_id": "travel", "trigger": "Travel",
                              "effects": "Teleport to target"}]
    assert not checked_rule_template_ready(spec, candidate, callbacks=callbacks)
