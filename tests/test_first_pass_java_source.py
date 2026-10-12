"""Regression checks for bounded first-pass Java source assembly."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.first_pass_java_source import (
    can_compose_first_pass_java,
    compose_first_pass_java,
    _assemble_java_imports,
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
                "imports": [],
                "superclass": "",
                "interfaces": "",
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
    assert "public class Gate {" in result
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


def test_host_writes_class_identity_without_requesting_model_declaration():
    router = FakeNativeRouter()
    result = compose_first_pass_java(router, SPEC, [])
    assert "package example.client;" in result
    assert "public class Gate {" in result
    parameters = router.calls[0][2]
    assert "declaration" not in parameters["properties"]
    assert all(name in parameters["properties"] for name in ("superclass", "interfaces"))


def test_superclass_and_interfaces_are_composed_by_host():
    class InterfaceRouter(FakeNativeRouter):
        def generate_tool_decision(self, *args, **kwargs):
            result = super().generate_tool_decision(*args, **kwargs)
            if kwargs["tool_name"] == "submit_java_construction_plan":
                return {**result, "superclass": "BaseScreen", "interfaces": "Runnable, AutoCloseable"}
            return result

    result = compose_first_pass_java(InterfaceRouter(), SPEC, [])
    assert "public class Gate extends BaseScreen implements Runnable, AutoCloseable {" in result


def test_multiline_type_clause_rejected_without_member_retry():
    class BadTypeRouter(FakeNativeRouter):
        def generate_tool_decision(self, *args, **kwargs):
            result = super().generate_tool_decision(*args, **kwargs)
            if kwargs["tool_name"] == "submit_java_construction_plan":
                return {**result, "superclass": "BaseScreen" + chr(10) + "public class Wrong"}
            return result

    router = BadTypeRouter()
    with pytest.raises(ValueError, match="JAVA_FIRST_PASS_TYPE_REFERENCE_INVALID"):
        compose_first_pass_java(router, SPEC, [])
    assert len(router.calls) == 1


def test_host_assembles_imports_from_references_and_existing_java_syntax():
    imports = _assemble_java_imports([
        "java.util.List",
        "java.util.Map",
        "import java.util.List;",
        "import net.minecraft.client.gui.screens.Screen; import java.util.Set;",
        "static java.util.Collections.emptyList",
        "net.minecraft.world.entity.*",
        "package unrelated;",
        "public class Oops {",
        "",
    ])
    assert imports.splitlines() == [
        "import java.util.List;",
        "import java.util.Map;",
        "import net.minecraft.client.gui.screens.Screen;",
        "import java.util.Set;",
        "import static java.util.Collections.emptyList;",
        "import net.minecraft.world.entity.*;",
    ]


def test_model_outputs_dependency_references_not_java_import_statements():
    class ImportRouter(FakeNativeRouter):
        def generate_tool_decision(self, *args, **kwargs):
            result = super().generate_tool_decision(*args, **kwargs)
            if kwargs["tool_name"] == "submit_java_construction_plan":
                return {
                    **result,
                    "imports": ["java.util.List", "java.util.Map"],
                }
            return result

    router = ImportRouter()
    java = compose_first_pass_java(router, SPEC, [])
    assert java.startswith(
        "package example.client;\n\nimport java.util.List;\n"
        "import java.util.Map;\n\npublic class Gate {"
    )
    parameters = router.calls[0][2]
    assert parameters["properties"]["imports"]["type"] == "array"
    assert len(router.calls) == 3


def test_invalid_import_suggestion_does_not_inject_code_or_trigger_model_retry():
    class NonImportRouter(FakeNativeRouter):
        def generate_tool_decision(self, *args, **kwargs):
            result = super().generate_tool_decision(*args, **kwargs)
            if kwargs["tool_name"] == "submit_java_construction_plan":
                return {**result, "imports": ["public class Injected {}", "java.util.List"]}
            return result

    router = NonImportRouter()
    result = compose_first_pass_java(router, SPEC, [])
    assert "public class Injected" not in result
    assert "import java.util.List;" in result
    assert [name for name, *_ in router.calls] == [
        "submit_java_construction_plan",
        "submit_java_member_1",
        "submit_java_member_2",
    ]
