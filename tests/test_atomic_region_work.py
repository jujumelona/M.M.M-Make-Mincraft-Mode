from __future__ import annotations

import json
from copy import deepcopy

import pytest

from minecraft_mod_ai.atomic_region_paging import decide_region_completion, generate_region
from minecraft_mod_ai.atomic_region_work import implements_target, target_contract, validate_schedule
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted


def _work(header, purpose="implement selected behavior"):
    return {"done": False, "next_work": purpose, "target": header}


DONE = {"done": True, "next_work": "", "target": ""}
FIELD = 'private static final String PLAYER_INVENTORY_SCOPE = "ShipParts";'
FIELD_HEADER = "private static final String PLAYER_INVENTORY_SCOPE"
METHOD = "public static String scope() { return PLAYER_INVENTORY_SCOPE; }"
METHOD_HEADER = "public static String scope()"


def _messages():
    return [
        {"role": "system", "content": "Implement Java concern."},
        {"role": "user", "content": json.dumps({
            "response_region": "members", "host_selected_class": "AuthoredPersistence",
            "concern": {"name": "serialization"},
            "task_authority": {"source_requirements": {"R1": "Return the ShipParts inventory scope."}},
        })},
    ]


class _Router:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        assert role == "coder"
        assert set(kwargs["parameters"]["required"]) == {"done", "next_work", "target"}
        self.calls.append(json.loads(messages[-1]["content"]))
        return deepcopy(next(self.decisions))


def _run(responses, decisions):
    responses = iter([OutputBudgetExhausted("OUTPUT_BUDGET_EXHAUSTED"), *responses])
    calls = []
    router = _Router(decisions)

    def coder(messages):
        calls.append(json.loads(messages[-1]["content"]))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    source = generate_region(coder, _messages(), completion_decider=lambda p: decide_region_completion(router, p))
    return source, calls, router.calls


def test_live_trace_echo_stays_on_selected_method_and_keeps_exact_field_values():
    source, calls, decisions = _run(
        [FIELD, FIELD, METHOD],
        [_work(FIELD_HEADER, "Store ShipParts as the scope"), _work(METHOD_HEADER), DONE],
    )
    assert source == FIELD + "\n\n" + METHOD
    assert len(decisions) == 3  # No completion call for the rejected echo.
    assert calls[2]["region_page"]["target"] == calls[3]["region_page"]["target"] == METHOD_HEADER
    assert calls[3]["page_correction"]["expected_target"] == METHOD_HEADER
    for call in calls[2:]:
        assert call["region_page"]["accepted_fields"] == [FIELD]
        assert call["region_page"]["completed_work"][0]["purpose"] == "Store ShipParts as the scope"
    assert len(decisions[-1]["completed_work"]) == 2


def test_native_selection_rejects_current_page_duplicate_before_source_generation():
    source, calls, decisions = _run(
        [FIELD, METHOD], [_work(FIELD_HEADER), _work(FIELD_HEADER), _work(METHOD_HEADER), DONE],
    )
    assert source.count(FIELD) == 1
    assert len(calls) == 3
    assert "TARGET_ALREADY_ACCEPTED" in decisions[2]["completion_feedback"]


def test_native_selection_rejects_earlier_page_duplicate_with_changed_purpose():
    router = _Router([_work(FIELD_HEADER, "repeat for a different purpose"), _work(METHOD_HEADER)])
    result = decide_region_completion(router, {
        "response_region": "members", "completion_phase": "select_next_unit",
        "accepted_api": [{"kind": "field", "symbol": "PLAYER_INVENTORY_SCOPE", "declared_type": "String"}],
    })
    assert result["target"] == METHOD_HEADER
    assert "TARGET_ALREADY_ACCEPTED" in router.calls[-1]["completion_feedback"]


def test_output_refinement_cannot_forget_parent_or_finish_after_only_a_helper():
    helper = "private static String readScope() { return PLAYER_INVENTORY_SCOPE; }"
    parent = "public static String scope() { return readScope(); }"
    source, calls, decisions = _run(
        [FIELD, OutputBudgetExhausted("OUTPUT_BUDGET_EXHAUSTED"), helper, parent],
        [_work(FIELD_HEADER), _work(METHOD_HEADER), _work("private static String readScope()"), DONE],
    )
    assert source == "\n\n".join([FIELD, helper, parent])
    assert calls[-1]["region_page"]["target"] == METHOD_HEADER
    assert decisions[-2]["deferred_work"][0]["target"] == METHOD_HEADER
    # The host resumes the parent without asking the model if the helper is enough.
    assert decisions[-1]["current_page_source"] == parent
    assert len(decisions[-1]["completed_work"]) == 3


def test_refinement_cannot_rename_the_exhausted_parent_as_a_smaller_task():
    router = _Router([_work(METHOD_HEADER, "smaller body"), _work("private static String readScope()")])
    result = decide_region_completion(router, {
        "response_region": "members", "completion_phase": "refine_next_unit",
        "deferred_work": [_work(METHOD_HEADER)],
    })
    assert result["target"] == "private static String readScope()"
    assert "TARGET_NOT_REFINED" in router.calls[-1]["completion_feedback"]


def test_helper_page_cannot_commit_a_deferred_parent_with_a_changed_api():
    helper_header = "private static String readScope()"
    helper = "private static String readScope() { return PLAYER_INVENTORY_SCOPE; }"
    bad_parent = "public static int scope() { return 1; }"
    source, calls, _ = _run(
        [FIELD, OutputBudgetExhausted("OUTPUT_BUDGET_EXHAUSTED"), helper + bad_parent, helper, METHOD],
        [_work(FIELD_HEADER), _work(METHOD_HEADER), _work(helper_header), DONE],
    )
    assert bad_parent not in source
    assert source == "\n\n".join([FIELD, helper, METHOD])
    assert calls[-2]["page_correction"]["conflicting_parent_targets"] == [METHOD_HEADER]


def test_helper_page_can_complete_parent_without_rescheduling_it():
    helper_header = "private static String readScope()"
    helper = "private static String readScope() { return PLAYER_INVENTORY_SCOPE; }"
    source, calls, decisions = _run(
        [FIELD, OutputBudgetExhausted("OUTPUT_BUDGET_EXHAUSTED"), helper + "\n" + METHOD],
        [_work(FIELD_HEADER), _work(METHOD_HEADER), _work(helper_header), DONE],
    )
    assert source == "\n\n".join([FIELD, helper, METHOD])
    assert len(calls) == 4
    assert not decisions[-1]["deferred_work"]
    assert {work["target"] for work in decisions[-1]["completed_work"]} == {
        FIELD_HEADER, METHOD_HEADER, helper_header,
    }


def test_wrong_target_does_not_commit_an_unrelated_valid_declaration():
    unrelated = "private static int unrelated = 8;"
    source, calls, _ = _run([unrelated, FIELD], [_work(FIELD_HEADER), DONE])
    assert source == FIELD
    assert not calls[-1]["region_page"]["completed_work"]
    assert calls[-1]["region_page"]["accepted_fields"] == []


def test_repeated_echo_is_bounded_and_never_becomes_completion():
    with pytest.raises(CustomModuleGenerationError, match="TARGET_MISSING"):
        _run([FIELD, FIELD, FIELD], [_work(FIELD_HEADER), _work(METHOD_HEADER)])


@pytest.mark.parametrize("header", [
    "", "private int a, b", "private int x = 4", "private int x;", "public void run() {}",
    "private static class A {} private static class B {}", "private void broken(",
])
def test_target_is_one_parsed_declaration_header(header):
    with pytest.raises(CustomModuleGenerationError, match="TARGET_INVALID"):
        target_contract(header)


@pytest.mark.parametrize("source", [
    "private static String scope() { return null; }",
    "public String scope() { return null; }",
    "public static int scope() { return 1; }",
    "public static String scope(int ignored) { return null; }",
])
def test_target_binding_preserves_visibility_static_return_and_parameter_contracts(source):
    assert not implements_target(METHOD_HEADER, source)


def test_target_binding_accepts_qualified_jdk_types_and_parameter_renaming():
    assert implements_target(
        "public static String echo(String input)",
        "public static java.lang.String echo(java.lang.String value) { return value; }",
    )
    assert implements_target(
        "public static String[] echo(String[] input)",
        "public static java.lang.String[] echo(java.lang.String[] values) { return values; }",
    )


def test_nested_record_target_keeps_component_contract():
    header = "private record Entry(String key, int value)"
    assert implements_target(header, header + " {}")
    assert not implements_target(header, "private record Entry(String key, long value) {}")


def test_host_cannot_finish_with_deferred_parent():
    with pytest.raises(CustomModuleGenerationError, match="unfinished parent"):
        validate_schedule(DONE, {"deferred_work": [_work(METHOD_HEADER)]})


def test_native_initialize_keeps_ordered_statement_contract():
    messages = _messages()
    payload = json.loads(messages[-1]["content"])
    payload["response_region"] = "initialize"
    messages[-1]["content"] = json.dumps(payload)
    responses = iter([OutputBudgetExhausted("OUTPUT_BUDGET_EXHAUSTED"), "int local = 1;", "value += local;"])
    router = _Router([_work("", "declare local"), _work("", "add local to value"), DONE])

    def coder(_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    source = generate_region(coder, messages, completion_decider=lambda p: decide_region_completion(router, p))
    assert source == "int local = 1;\n\nvalue += local;"
    assert router.calls[-1]["accepted_source"] == "int local = 1;"
