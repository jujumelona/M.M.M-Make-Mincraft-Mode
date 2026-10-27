"""Archived role confusion cannot cross the structured data/code boundary."""
from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
from minecraft_mod_ai.incremental_java_model import initial_scope
from minecraft_mod_ai.incremental_java_operations import author_operation


def test_archived_method_type_and_declaration_response_are_not_accepted_value_pages():
    forbidden_type = {"type": "method"}
    forbidden_value = {"implementation": "privatevoidmmmHelper3()//TODO:implement"}
    class Coder:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            task = json.loads(messages[-1]["content"])
            page = task["requested_page"]
            validator = Draft202012Validator(parameters)
            assert not validator.is_valid(forbidden_type)
            assert not validator.is_valid(forbidden_value)
            if page == "local":
                return {"type": "int"}
            if page == "value_kind":
                return {"kind": "literal"}
            return {"type": "int"} if page == "literal_type" else {"value": 7}

        def generate_java_source(self, *args, **kwargs):
            raise AssertionError("The feature value path still asks the model for Java source")
    spec = {"target_path": "src/main/java/demo/Feature.java", "bindings": {"package_name": "demo", "class_name": "Feature"}}
    builder = IncrementalJavaBuilder(Coder(), spec, [], {"symbols": {}}, [{"behavior_id": "example", "intent": "Compute a value"}])
    builder.member = builder.program["members"][0]
    builder.scope = initial_scope(builder.member)
    node = author_operation(builder, "local")
    assert node["type"] == "int" and node["value"] == "7"


def test_code_like_literal_data_is_not_mistaken_for_an_unimplemented_method():
    from minecraft_mod_ai.construction_file_author import _check_identity
    source = 'package demo; public final class Feature { String message="privatevoidmmmHelper3()//TODO:implement UnsupportedOperationException"; }'
    _check_identity(source, "demo", "Feature")


@pytest.mark.parametrize("body", ["// TODO implement\nint count;", "void call(){throw new UnsupportedOperationException();}"])
def test_actual_unimplemented_comments_and_exception_stubs_are_still_detected(body):
    from minecraft_mod_ai.construction_file_author import _check_identity, ConstructionFileError
    with pytest.raises(ConstructionFileError, match="UNIMPLEMENTED_PLACEHOLDER"):
        _check_identity("package demo; public final class Feature {" + body + "}", "demo", "Feature")
