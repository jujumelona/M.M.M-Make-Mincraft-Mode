"""Runtime boundaries for literals, generated types, and intrinsic members."""
from __future__ import annotations

import pytest

from test_incremental_java_values import builder, execute, literal, local, member_named, symbols


def test_string_data_exceeding_one_classfile_utf8_constant_is_still_renderable(tmp_path):
    text = "x" * 70000
    value = builder(literal(text, "java.lang.String")).expression("value", "java.lang.String")
    execute(tmp_path, value, expected='"x".repeat(70000)', java_type="java.lang.String")


def test_string_null_unicode_and_backslash_unicode_escape_stay_data(tmp_path):
    text = "before\0after\\u000a\U0001f600"
    value = builder(literal(text, "java.lang.String")).expression("value", "java.lang.String")
    execute(tmp_path, value, expected='"before" + (char)0 + "after\\\\u000a😀"', java_type="java.lang.String")


def test_static_indexed_field_executes(tmp_path):
    owner = "java.lang.Integer"
    api = symbols(owner, {"kind": "FIELD", "owner": owner, "name": "MAX_VALUE", "access": 25,
                          "return_type": "int"})
    steps = [("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "static"}),
             ("static_owner", {"java_type": owner}), ("field_member", member_named("MAX_VALUE"))]
    execute(tmp_path, builder(steps, symbols=api).expression("value", "int"), expected="Integer.MAX_VALUE")


def test_generated_type_implicit_constructor_is_selected_by_type_id(tmp_path):
    declared = {"id": "t1", "kind": "class", "name": "MmmType1", "owner": "Feature", "body": [],
                "metadata": {"static": True}}
    steps = [("value_kind", {"kind": "new"}), ("new_type", {"java_type": "MmmType1"}),
             ("constructor_member", {"member_id": "t1"})]
    value = builder(steps, members=[declared]).expression("value", "MmmType1")
    execute(tmp_path, "(" + value + ").getClass()", expected="MmmType1.class", java_type="Class<?>",
            declarations="static class MmmType1 {}")


def test_new_parameterized_type_retains_actual_constructor_signature(tmp_path):
    owner = "java.util.ArrayList"
    api = symbols(owner, {"kind": "CONSTRUCTOR", "owner": owner, "name": "<init>", "parameters": ["int"],
                          "access": 1}, type_parameters=["E"])
    steps = [("value_kind", {"kind": "new"}),
             ("new_type", {"java_type": "java.util.ArrayList<java.lang.String>"}),
             ("constructor_member", member_named("<init>")), *literal(12)]
    value = builder(steps, symbols=api).expression("value", "java.util.List<java.lang.String>")
    execute(tmp_path, "(" + value + ").size()", expected="0")


def test_multidimensional_array_supports_partial_sized_dimensions(tmp_path):
    steps = [("value_kind", {"kind": "array"}), ("array_type", {"java_type": "int[][]"}),
             ("array_mode", {"mode": "sizes"}), ("array_dimensions_count", {"count": 1}), *literal(12)]
    value = builder(steps).expression("value", "int[][]")
    execute(tmp_path, value + ".length")


def test_array_constructor_reference_executes(tmp_path):
    steps = [("value_kind", {"kind": "method_reference"}),
             ("reference_kind", {"reference_kind": "constructor"}),
             ("reference_type", {"java_type": "int[]"})]
    value = builder(steps).expression("value", "java.util.function.IntFunction<int[]>")
    execute(tmp_path, "((java.util.function.IntFunction<int[]>)" + value + ").apply(12).length")


def test_intrinsic_array_length_cannot_be_selected_for_assignment():
    def no_writable_length(schema, task):
        assert "array:length" not in task["member_candidates"]
        raise RuntimeError("read only")

    steps = [("value_kind", {"kind": "field"}), ("field_receiver", {"receiver_kind": "value"}),
             *local("array"), ("field_member", no_writable_length)]
    result = builder(steps, locals=[{"name": "array", "type": "int[]"}])
    with pytest.raises(RuntimeError, match="read only"):
        result.expression("target")
    assert result.value_stack == []


def test_literal_type_page_only_offers_scalar_types():
    def inspect_types(schema, task):
        choices = schema["properties"]["java_type"]["enum"]
        assert "java.lang.Object" not in choices
        assert not any(value.endswith("[]") or "<" in value for value in choices)
        return {"java_type": "int"}

    result = builder([("value_kind", {"kind": "literal"}), ("literal_type", inspect_types),
                      ("literal", {"value": 12})])
    assert result.expression("value", "java.lang.Object") == "12"


def test_small_float_payload_rounds_to_valid_java_float(tmp_path):
    value = builder(literal(1e-50, "float")).expression("value", "float")
    execute(tmp_path, value, expected="0.0f", java_type="float")


@pytest.mark.parametrize("java_type", ["char", "java.lang.String"])
def test_unpaired_utf16_unit_is_preserved_without_invalid_source(tmp_path, java_type):
    value = builder(literal("\ud800", java_type)).expression("value", java_type)
    expected = "(char)55296" if java_type == "char" else "java.lang.String.valueOf((char)55296)"
    execute(tmp_path, value, expected=expected, java_type=java_type)


def test_primitive_class_literal_reports_boxed_class_type():
    result = builder([("value_kind", {"kind": "class_literal"}),
                      ("class_literal_type", {"java_type": "int"})])
    assert result.expression("value", "java.lang.Class") == "int.class"
    assert result.last_value_type == "java.lang.Class<java.lang.Integer>"
