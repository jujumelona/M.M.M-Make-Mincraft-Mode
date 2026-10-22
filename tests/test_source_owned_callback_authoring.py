"""A feature without an inspected occurrence cannot be 'implemented' by no-arg code."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
from minecraft_mod_ai.source_owned_hook import SourceHookUnavailable


def test_source_owned_feature_requires_inspected_runtime_occurrence(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces",
        lambda symbols, side: [],
    )

    class NoUnsafeCoder:
        def generate_structured_data(self, *args, **kwargs):
            raise AssertionError("do not generate an arbitrary callback/registrar")

        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("no unchecked Java source must be authored")

    spec = {
        "target_path": "src/main/java/example/Callback.java", "language": "java",
        "requirement": "Perform world interaction",
        "bindings": {
            "package_name": "example", "class_name": "Callback",
            "request_feature_entrypoint": True,
        },
    }
    behaviors = [{
        "behavior_id": "generic", "intent": "Interact with world",
        "source_quote": "Interact with world", "effects": "Real world interaction",
    }]
    with pytest.raises((SourceHookUnavailable, ValueError), match=(
            "SOURCE_HOOK_PRODUCER_REQUIRED|JAVA_HOST_FABRIC_EVENT_ABI_UNAVAILABLE")):
        compose_first_pass_java(
            NoUnsafeCoder(), spec, [], behaviors=behaviors,
            construction_context={
                "symbols": {}, "authoring_contract": {},
                "host_source_owned": True,
            })
