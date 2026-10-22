"""Source-owned execution never re-enters the unsuccessful hook-selection model."""
from __future__ import annotations

import hashlib
import json

from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java


def test_source_owned_feature_authors_bounded_registrar_without_model_event_picker(monkeypatch):
    calls = []
    def no_hook_search(*args, **kwargs):
        raise AssertionError("source ownership cannot return to the failed native event pager")
    monkeypatch.setattr(
        "minecraft_mod_ai.host_fabric_callback_abi.callback_interfaces",
        no_hook_search,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.host_java_imports.host_add_verified_imports",
        lambda source, **kwargs: source,
    )

    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            calls.append(data_name)
            if data_name.startswith("submit_java_declaration_"):
                return {
                    "visibility": "private", "static": False,
                    "return_type": "void", "type_parameters": "",
                    "parameters": [], "throws": [], "annotations": [],
                }
            raise AssertionError(data_name)

        def generate_java_source(self, role, messages, *, data_name, source_frame):
            task = json.loads(messages[-1]["content"])
            calls.append((data_name, task["member_role"]))
            if task["member_role"] == "register_callback":
                # Real source implementation is separately javac-validated.
                body = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents.END_SERVER_TICK.register(s -> {});"
            else:
                body = "System.out.println(\"implemented\");"
            return source_frame["prefix"] + body + source_frame["suffix"]

    behavior = {
        "behavior_id": "behavior_generic", "intent": "Perform real world interaction",
        "source_quote": "Perform real world interaction",
        "trigger": "", "effects": "World interaction",
    }
    spec = {
        "target_path": "src/main/java/example/Callback.java", "language": "java",
        "requirement": "Perform real world interaction",
        "bindings": {"package_name": "example", "class_name": "Callback",
                     "request_feature_entrypoint": True},
    }
    output = compose_first_pass_java(
        Coder(), spec, [], behaviors=[behavior],
        construction_context={
            "symbols": {}, "authoring_contract": {},
            "host_source_owned": True,
        },
    )
    registrar = "registerBehavior" + hashlib.sha256(b"behavior_generic").hexdigest()[:20]
    assert "implements net.fabricmc.api.ModInitializer" in output
    assert registrar + "()" in output
    assert "END_SERVER_TICK.register(" in output
    assert ("submit_java_implementation_2", "register_callback") in calls
