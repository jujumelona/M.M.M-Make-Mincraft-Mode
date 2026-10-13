"""Contracts for paged Java planning, typed declarations and fixed-source frames.

The fake records the production router interface. These are regression fixtures,
not a second, legacy Java-generation protocol.
"""
from __future__ import annotations

import re

import pytest

from minecraft_mod_ai.first_pass_java_source import (
    _assemble_java_imports,
    can_compose_first_pass_java,
    compose_first_pass_java,
)


SPEC = {
    "language": "java",
    "target_path": "src/main/java/example/client/Gate.java",
    "bindings": {"package_name": "example.client", "class_name": "Gate"},
    "requirement": "Create an operational readiness gate.",
}


class RecordingJavaRouter:
    """Answer each typed plan page, then complete its immutable source frame."""

    def __init__(self, *, imports=(), superclass="", interfaces="", units=None,
                 fail_on=None):
        self.imports = list(imports)
        self.superclass = superclass
        self.interfaces = interfaces
        self.units = list(units if units is not None else [
            {"kind": "method", "name": "isReady", "purpose": "Expose readiness"}
        ])
        self.fail_on = fail_on
        self.data_calls = []
        self.source_calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.data_calls.append((data_name, parameters))
        if self.fail_on == data_name:
            raise RuntimeError("synthetic model failure")
        assert role == "coder"
        # No obsolete per-call output_token_floor/ceiling override: the
        # centralized model transport budget owns those limits.
        properties = parameters["properties"]
        assert parameters["additionalProperties"] is False
        if data_name.startswith("submit_java_construction_plan_"):
            page = data_name.removeprefix("submit_java_construction_plan_")
            if page == "imports_count":
                return {"count": len(self.imports)}
            if page == "units_count":
                return {"count": len(self.units)}
            match = re.fullmatch(r"imports_(\d+)", page)
            if match:
                return {"value": self.imports[int(match.group(1)) - 1]}
            match = re.fullmatch(r"units_(\d+)_value_(kind|name|purpose)", page)
            if match:
                return {match.group(2): self.units[int(match.group(1)) - 1][match.group(2)]}
            if page == "superclass":
                return {"superclass": self.superclass}
            if page == "interfaces":
                return {"interfaces": self.interfaces}
        if data_name == "submit_java_declaration_1":
            assert "name" not in properties
            assert "body" not in properties
            return {
                "visibility": "public",
                "static": False,
                "return_type": "boolean",
                "type_parameters": "",
                "parameters": [],
                "throws": [],
                "annotations": [],
            }
        raise AssertionError("Unexpected first-pass Java data page: " + data_name)

    def generate_java_source(self, role, messages, *, data_name, source_frame):
        self.source_calls.append((data_name, source_frame))
        if self.fail_on == data_name:
            raise RuntimeError("synthetic model failure")
        assert role == "coder"
        assert source_frame["kind"] == "method"
        if data_name == "submit_java_implementation_1":
            return source_frame["prefix"] + "    return true;" + source_frame["suffix"]
        if data_name == "submit_java_entrypoint_source":
            return source_frame["prefix"] + "    // Host-owned lifecycle declaration." + source_frame["suffix"]
        raise AssertionError("Unexpected Java implementation action: " + data_name)


def _data_names(router):
    return [name for name, _ in router.data_calls]


def _source_names(router):
    return [name for name, _ in router.source_calls]


def test_first_pass_builds_complete_java_from_typed_metadata_and_fixed_source():
    router = RecordingJavaRouter()
    assert can_compose_first_pass_java(SPEC)
    source = compose_first_pass_java(router, SPEC, [], source_context=())
    assert source.startswith("package example.client;\n\npublic class Gate {")
    assert source.count("public boolean isReady() {") == 1
    assert source.count("return true;") == 1
    assert source.rstrip().endswith("}")
    assert _data_names(router) == [
        "submit_java_construction_plan_imports_count",
        "submit_java_construction_plan_superclass",
        "submit_java_construction_plan_interfaces",
        "submit_java_construction_plan_units_count",
        "submit_java_construction_plan_units_1_value_kind",
        "submit_java_construction_plan_units_1_value_name",
        "submit_java_construction_plan_units_1_value_purpose",
        "submit_java_declaration_1",
    ]
    assert _source_names(router) == ["submit_java_implementation_1"]
    assert router.source_calls[0][1]["prefix"] == "public boolean isReady() {\n"


def test_failed_source_is_never_replayed_or_regenerated():
    router = RecordingJavaRouter(fail_on="submit_java_implementation_1")
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        compose_first_pass_java(router, SPEC, [])
    assert _source_names(router) == ["submit_java_implementation_1"]
    assert _data_names(router).count("submit_java_declaration_1") == 1


def test_failed_metadata_is_not_retried_as_raw_java():
    router = RecordingJavaRouter(fail_on="submit_java_declaration_1")
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        compose_first_pass_java(router, SPEC, [])
    assert _data_names(router).count("submit_java_declaration_1") == 1
    assert router.source_calls == []


def test_fragment_target_does_not_claim_a_complete_class_boundary():
    assert not can_compose_first_pass_java({
        **SPEC, "java_prefix": "public class Surrounding {",
    })


def test_target_identity_must_match_declared_source_path():
    assert not can_compose_first_pass_java({
        **SPEC,
        "bindings": {**SPEC["bindings"], "class_name": "Other"},
    })


def test_host_owns_class_identity_and_model_only_selects_member_metadata():
    router = RecordingJavaRouter()
    source = compose_first_pass_java(router, SPEC, [])
    assert "package example.client;" in source
    assert "public class Gate {" in source
    assert all("declaration" not in schema["properties"]
               for _, schema in router.data_calls)
    assert "name" not in dict(router.data_calls)["submit_java_declaration_1"]["properties"]


def test_superclass_interfaces_and_imports_are_rendered_by_host():
    router = RecordingJavaRouter(
        imports=["java.util.List", "java.util.Map"],
        superclass="BaseScreen",
        interfaces="Runnable, AutoCloseable",
    )
    source = compose_first_pass_java(router, SPEC, [])
    assert source.startswith(
        "package example.client;\n\nimport java.util.List;\n"
        "import java.util.Map;\n\npublic class Gate "
    )
    assert "extends BaseScreen implements Runnable, AutoCloseable {" in source
    assert _data_names(router)[:3] == [
        "submit_java_construction_plan_imports_count",
        "submit_java_construction_plan_imports_1",
        "submit_java_construction_plan_imports_2",
    ]


def test_invalid_superclass_clause_is_rejected_before_member_generation():
    router = RecordingJavaRouter(superclass="BaseScreen\npublic class Wrong")
    with pytest.raises(ValueError, match="JAVA_FIRST_PASS_TYPE_REFERENCE_INVALID"):
        compose_first_pass_java(router, SPEC, [])
    assert router.source_calls == []
    assert "submit_java_declaration_1" not in _data_names(router)


def test_host_assembles_only_valid_deduplicated_import_references():
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


def test_invalid_import_suggestion_cannot_inject_java_or_trigger_retry():
    router = RecordingJavaRouter(imports=["public class Injected {}", "java.util.List"])
    source = compose_first_pass_java(router, SPEC, [])
    assert "public class Injected" not in source
    assert "import java.util.List;" in source
    assert _data_names(router).count("submit_java_construction_plan_imports_count") == 1
    assert _source_names(router) == ["submit_java_implementation_1"]


def test_client_entrypoint_is_host_owned_and_not_redeclared_by_model():
    router = RecordingJavaRouter()
    client = {**SPEC, "side": "CLIENT"}
    source = compose_first_pass_java(router, client, [])
    assert "public final class Gate implements net.fabricmc.api.ClientModInitializer {" in source
    assert "boolean mmmMember1()" in source
    assert source.count("void onInitializeClient()") == 1
    assert _source_names(router) == [
        "submit_java_implementation_1",
        "submit_java_entrypoint_source",
    ]
    assert "submit_java_construction_plan_superclass" not in _data_names(router)
    assert "submit_java_construction_plan_interfaces" not in _data_names(router)
