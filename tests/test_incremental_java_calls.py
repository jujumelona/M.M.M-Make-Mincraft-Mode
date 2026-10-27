"""Arguments follow exact declarations without gating newly authored gameplay."""
from __future__ import annotations

from copy import deepcopy

import pytest


class CallBuilder:
    def __init__(self, *, symbols=None, members=(), locals=(), answers=()):
        from minecraft_mod_ai.incremental_java_context import IncrementalJavaContext

        spec = {"target_path": "src/main/java/demo/Feature.java",
                "bindings": {"package_name": "demo", "class_name": "Feature"}}
        self.api = IncrementalJavaContext([], spec, {"symbols": symbols or {}}, [])
        self.program = {"package": "demo", "class_name": "Feature", "members": list(members)}
        self.member = {"owner": "Feature", "kind": "method", "name": "install"}
        self.scope = {"locals": list(locals)}
        self.answers = iter(answers)
        self.requests, self.expressions, self.argument_prefixes = [], [], []
        self.drafts = []

    def ask(self, schema, suffix, extra=None):
        from minecraft_mod_ai.fixed_template_generation import _validate_native_arguments

        self.requests.append((deepcopy(schema), suffix, deepcopy(extra)))
        answer = next(self.answers)
        if callable(answer):
            answer = answer(schema, extra)
        return _validate_native_arguments(answer, schema)

    def expression(self, slot, expected="java.lang.Object"):
        self.expressions.append((slot, expected))
        self.argument_prefixes.append(list(self.drafts[-1]["arguments"]))
        return "value" + str(len(self.expressions) - 1)


def _method(owner, name, parameters=(), **extra):
    return {"kind": "METHOD", "owner": owner, "name": name,
            "parameters": list(parameters), "return_type": "void", "access": 1, **extra}


def _symbols(owner, *methods, **extra):
    rows = {owner: {"kind": "CLASS", "owner": owner, "name": owner.rsplit(".", 1)[-1],
                    "access": 1, **extra}}
    rows.update({"method" + str(index): method for index, method in enumerate(methods)})
    return rows


def _author(builder, node):
    from minecraft_mod_ai.incremental_java_calls import author_call_arguments

    builder.drafts.append(node)
    try:
        author_call_arguments(builder, node)
    finally:
        builder.drafts.pop()
    return node["arguments"]


@pytest.mark.parametrize("receiver", ["", "this", "Feature", "demo.Feature"])
def test_new_two_argument_helper_is_callable_before_its_body_without_a_count_turn(receiver):
    helper = {"kind": "method", "owner": "Feature", "name": "mmmHelper2", "body": [],
              "metadata": {"parameters": [{"type": "example.Player", "name": "arg0"},
                                            {"type": "long", "name": "arg1"}],
                           "return_type": "void"}}
    builder = CallBuilder(members=[helper])

    assert _author(builder, {"op": "call", "receiver": receiver, "method": "mmmHelper2"}) == ["value0", "value1"]

    assert builder.requests == []
    assert builder.expressions == [("argument:0", "example.Player"), ("argument:1", "long")]
    assert builder.argument_prefixes == [[], ["value0"]]


def test_generic_receiver_set_uses_specialized_source_parameter_types():
    owner = "java.util.List"
    symbols = _symbols(owner, _method(owner, "set", ["int", "java.lang.Object"],
                                      generic_parameters=["int", "E"],
                                      generic_return_type="E"), type_parameters=["E"])
    builder = CallBuilder(symbols=symbols, locals=[{"name": "values", "type": "java.util.List<java.lang.String>"}])

    _author(builder, {"op": "call", "receiver": "values", "method": "set"})

    assert builder.requests == []
    assert builder.expressions == [("argument:0", "int"), ("argument:1", "java.lang.String")]


@pytest.mark.parametrize("receiver", ["Tools", "example.Tools"])
def test_class_receiver_uses_exact_static_api_and_preserves_nested_wildcards(receiver):
    owner = "example.Tools"
    symbols = _symbols(owner, _method(owner, "accept", ["java.util.List"], static=True, access=9,
                                      generic_parameters=["java.util.List<? extends example.Player>"]))
    builder = CallBuilder(symbols=symbols)

    _author(builder, {"op": "call", "receiver": receiver, "method": "accept"})

    assert builder.requests == []
    assert builder.expressions == [("argument:0", "java.util.List<? extends example.Player>")]


def test_generated_field_receiver_uses_its_parameterized_type():
    owner = "java.util.List"
    symbols = _symbols(owner, _method(owner, "set", ["int", "java.lang.Object"],
                                      generic_parameters=["int", "E"]), type_parameters=["E"])
    field = {"kind": "field", "owner": "Feature", "name": "mmmField1",
             "metadata": {"type": "java.util.List<java.lang.String>"}}
    builder = CallBuilder(symbols=symbols, members=[field])

    _author(builder, {"op": "call", "receiver": "this.mmmField1", "method": "set"})

    assert builder.expressions == [("argument:0", "int"), ("argument:1", "java.lang.String")]


@pytest.mark.parametrize("count, expected", [
    (1, [("argument:0", "int")]),
    (3, [("argument:0", "int"), ("argument:1", "java.lang.String"), ("argument:2", "java.lang.String")]),
])
def test_varargs_count_includes_fixed_parameters_and_expands_generic_components(count, expected):
    owner = "example.Logger"
    symbols = _symbols(owner, _method(owner, "emit", ["int", "java.lang.Object[]"],
                                      generic_parameters=["int", "java.lang.String[]"], varargs=True, access=0x81))
    builder = CallBuilder(symbols=symbols, locals=[{"name": "logger", "type": owner}], answers=[{"count": count}])

    _author(builder, {"op": "call", "receiver": "logger", "method": "emit"})

    schema, suffix, _ = builder.requests[0]
    assert suffix == "arguments_count"
    assert schema["properties"]["count"] == {"type": "integer", "minimum": 1, "maximum": 255}
    assert builder.expressions == expected


def test_late_overload_page_is_selectable_without_exposing_other_method_names():
    owner = "example.Many"
    methods = [_method(owner, "apply", ["example.Input" + str(index)]) for index in range(23)]
    methods.append(_method(owner, "unrelatedGameplay", ["int"]))
    symbols = _symbols(owner, *methods)

    def next_page(schema, extra):
        assert len(extra["overload_candidates"]) == 10
        assert {row["name"] for row in extra["overload_candidates"].values()} == {"apply"}
        assert "next_page" in schema["properties"]["overload_id"]["enum"]
        return {"overload_id": "next_page"}

    def choose_last(schema, extra):
        assert len(extra["overload_candidates"]) == 3
        identity = next(key for key, row in extra["overload_candidates"].items()
                        if row["parameters"] == ["example.Input9"])
        return {"overload_id": identity}

    builder = CallBuilder(symbols=symbols, locals=[{"name": "many", "type": owner}],
                          answers=[next_page, next_page, choose_last])

    _author(builder, {"op": "call", "receiver": "many", "method": "apply"})

    assert len(builder.requests) == 3
    assert builder.expressions == [("argument:0", "example.Input9")]


@pytest.mark.parametrize("receiver", ["this", "missingValue", "example.Missing"])
def test_unknown_gameplay_method_falls_back_to_count_and_individual_expressions(receiver):
    owner = "example.Unrelated"
    symbols = _symbols(owner, _method(owner, "newGameplay", ["int"]))
    builder = CallBuilder(symbols=symbols, answers=[{"count": 2}])

    assert _author(builder, {"op": "call", "receiver": receiver, "method": "newGameplay"}) == ["value0", "value1"]

    assert len(builder.requests) == 1
    assert builder.requests[0][1] == "arguments_count"
    assert builder.expressions == [("argument:0", "java.lang.Object"), ("argument:1", "java.lang.Object")]


def test_override_only_protected_methods_are_not_offered_as_callable_overloads():
    owner = "example.Hidden"
    symbols = _symbols(owner, _method(owner, "transform", ["int"], access=4))
    builder = CallBuilder(symbols=symbols, locals=[{"name": "hidden", "type": owner}], answers=[{"count": 2}])

    assert _author(builder, {"op": "call", "receiver": "hidden", "method": "transform"}) == ["value0", "value1"]

    assert builder.requests[0][1] == "arguments_count"
    assert builder.expressions == [("argument:0", "java.lang.Object"), ("argument:1", "java.lang.Object")]


def test_constructor_call_retains_unrestricted_argument_count_fallback():
    builder = CallBuilder(answers=[{"count": 2}])

    _author(builder, {"op": "constructor_call", "target": "super"})

    assert builder.requests[0][0]["properties"]["count"]["maximum"] == 255
    assert builder.expressions == [("argument:0", "java.lang.Object"), ("argument:1", "java.lang.Object")]


@pytest.mark.parametrize("source_type, expected", [
    ("?", "java.lang.Object"),
    ("? extends example.Value", "example.Value"),
    ("? super example.Value", "example.Value"),
])
def test_top_level_wildcards_are_grounded_for_expression_field_frames(source_type, expected):
    owner = "example.Consumer"
    symbols = _symbols(owner, _method(owner, "accept", ["java.lang.Object"], generic_parameters=[source_type]))
    builder = CallBuilder(symbols=symbols, locals=[{"name": "consumer", "type": owner}])

    _author(builder, {"op": "call", "receiver": "consumer", "method": "accept"})

    assert builder.expressions == [("argument:0", expected)]
