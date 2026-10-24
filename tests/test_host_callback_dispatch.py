import pytest

from minecraft_mod_ai.host_callback_dispatch import (
    render_callback_dispatch,
    select_callback_results,
)
from minecraft_mod_ai.structured_output import StructuredOutputValidationError


class ResultRouter:
    def __init__(self, values):
        self.values = iter(values)
        self.schemas = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.schemas.append(parameters)
        return {"expression": next(self.values)}


def test_host_result_choices_include_real_subtype_fields_and_exclude_other_fields():
    result = "example.Result"
    index = {
        "class": {"kind": "CLASS", "owner": "example.Success", "interfaces": [result]},
        "pass": {"kind": "FIELD", "owner": result, "name": "PASS", "return_type": result, "access": 25},
        "ok": {"kind": "FIELD", "owner": result, "name": "SUCCESS", "return_type": "example.Success", "access": 25},
        "wrong": {"kind": "FIELD", "owner": result, "name": "NUMBER", "return_type": "int", "access": 25},
        "private": {"kind": "FIELD", "owner": result, "name": "SECRET", "return_type": result, "access": 26},
    }
    router = ResultRouter([result + ".PASS", result + ".SUCCESS"])
    policy = select_callback_results(router, {"return_type": result}, index)
    assert policy == {"unhandled": result + ".PASS", "handled": result + ".SUCCESS"}
    assert router.schemas[0]["properties"]["expression"]["enum"] == [result + ".PASS", result + ".SUCCESS"]


def test_void_callback_does_not_invoke_model_for_return_values():
    assert select_callback_results(None, {"return_type": "void"}, {}) == {}


def test_inspected_boolean_callback_chooses_only_boolean_values():
    router = ResultRouter(["true", "false"])
    assert select_callback_results(router, {"return_type": "boolean"}, {}) == {
        "unhandled": "true", "handled": "false"}


def test_result_selection_cannot_smuggle_return_statement():
    with pytest.raises((ValueError, StructuredOutputValidationError)):
        select_callback_results(ResultRouter(["return true;"]), {"return_type": "boolean"}, {})


def test_nonvoid_dispatch_checks_condition_before_effect_and_handled_result():
    body = render_callback_dispatch("matches", "apply", "target", {
        "unhandled": "example.Result.PASS", "handled": "example.Result.SUCCESS"})
    assert body.index("if (!this.matches(target))") < body.index("return example.Result.PASS;")
    assert body.index("this.apply(target)") < body.index("return example.Result.SUCCESS;")
    assert "if (!this.apply(target))" in body


def test_predicate_and_effect_return_values_are_not_confused():
    assert render_callback_dispatch("matches", "apply", "x", {}) == (
        "if (this.matches(x)) {\n    this.apply(x);\n}")


def test_colonization_observers_gate_runtime_effect_before_dispatch():
    names = ["observeRuntimeaabbccdd", "observeRuntimeeeff0011"]
    text = render_callback_dispatch(
        "matchesColonization", "applyColonization", "player, level", {},
        observers=names,
    )
    assert text.startswith("if (this.matchesColonization(player, level) && ")
    assert "this.observeRuntimeaabbccdd(player, level)" in text
    assert "this.observeRuntimeeeff0011(player, level)" in text
    assert text.index("observeRuntimeeeff0011") < text.index("applyColonization")
    with pytest.raises(ValueError, match="DUPLICATE"):
        render_callback_dispatch("matches", "apply", "player", {},
                                 observers=["matches"])


def test_nonvoid_observer_failure_returns_unhandled_not_success():
    text = render_callback_dispatch(
        "matches", "apply", "world",
        {"unhandled": "false", "handled": "true"},
        observers=["observePrerequisites"],
    )
    assert "if (!(this.matches(world) && this.observePrerequisites(world)))" in text
    assert text.index("return false;") < text.index("this.apply(world)")
    assert text.endswith("return true;")
