"""Regression checks for bounded first-pass Java source assembly."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.first_pass_java_source import (
    can_compose_first_pass_java,
    compose_first_pass_java,
)


SPEC = {
    "language": "java",
    "target_path": "src/client/java/example/client/Gate.java",
    "bindings": {
        "package_name": "example.client",
        "class_name": "Gate",
        "minecraft_version": "26.2",
    },
    "requirement": "Create an operational readiness gate.",
}


class FakeNativeRouter:
    def __init__(self, *, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def generate_tool_decision(self, role, messages, *, tool_name, parameters,
                               description="", output_token_ceiling=None,
                               force_non_thinking=False):
        self.calls.append((tool_name, output_token_ceiling, parameters))
        if self.fail_on == tool_name:
            raise RuntimeError("synthetic model failure")
        answers = {
            "submit_java_construction_plan": {
                "imports": "",
                "declaration": "public final class Gate {",
                "units": [
                    {"name": "state", "purpose": "Store readiness state"},
                    {"name": "check", "purpose": "Expose readiness result"},
                ],
            },
            "submit_java_member_1": {
                "member": "private final boolean ready = true;"
            },
            "submit_java_member_2": {
                "member": "public boolean isReady() { return ready; }"
            },
        }
        return answers[tool_name]


def test_first_pass_builds_complete_java_from_small_member_calls():
    router = FakeNativeRouter()
    assert can_compose_first_pass_java(SPEC)
    result = compose_first_pass_java(router, SPEC, [], source_context=())
    assert result.startswith("package example.client;")
    assert "public final class Gate {" in result
    assert result.count("private final boolean ready = true;") == 1
    assert result.count("public boolean isReady()") == 1
    assert result.rstrip().endswith("}")
    assert [name for name, _, _ in router.calls] == [
        "submit_java_construction_plan",
        "submit_java_member_1",
        "submit_java_member_2",
    ]
    assert [budget for _, budget, _ in router.calls] == [1536, 2304, 2304]


def test_failed_member_is_not_retried_or_regenerated():
    router = FakeNativeRouter(fail_on="submit_java_member_1")
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        compose_first_pass_java(router, SPEC, [], source_context=())
    assert [name for name, _, _ in router.calls] == [
        "submit_java_construction_plan", "submit_java_member_1"
    ]


def test_fragment_target_does_not_claim_a_complete_class_boundary():
    assert not can_compose_first_pass_java({
        **SPEC, "java_prefix": "public class Surrounding {",
    })


def test_target_identity_must_match_declared_source_path():
    assert not can_compose_first_pass_java({
        **SPEC,
        "bindings": {**SPEC["bindings"], "class_name": "Other"},
    })
