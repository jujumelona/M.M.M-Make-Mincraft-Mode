from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.atomic_concern_source import (
    AtomicConcernExecutor,
    _deterministic_state_variable_members,
    _state_variable_contract,
)
from minecraft_mod_ai.atomic_java_admission import _semantic_component_issue
from minecraft_mod_ai.authored_ir_parser import slice_concern_requirements
from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
from minecraft_mod_ai.implementation_graph_execution import _bind_atomic_leaf_contract


STATE_REQUIREMENTS = {
    "R29": "## state_model",
    "R30": "- variables: name owner type unit default domain",
    "R31": "  - **PlayerBalance**: 소유자 `Player`, 타입 `double`, 단위 `Credits`, 디폴트 `0.0`, 도메인 `EconomySystem`.",
    "R32": "  - **ShipComponents**: 소유자 `Entity`, 타입 `List`, 단위 `Blueprints`, 디폴트 `[]`, 도메인 `CraftingTable`.",
    "R33": "  - **PlanetControl`: 소유자 `WorldManager`, 타입 `EnumSet`, 단위 `Colonies`, 디폴트 `empty_set`, 도메인 `SpaceMap`.",
    "R34": "- transitions: from_state trigger guard mutation to_state",
    "R35": "  - **Idle -> Charging**: Trigger: `FuelingItemUsed`, Guard: `fuel_capacity > current_fuel`, Mutation: `fuel_current += item.amount`, To: `ChargingState`.",
    "R36": "  - **Mining -> Selling**: Trigger: `MerchantInteract`, Guard: `player_credits < cost`, Mutation: `balance -= cost`, To: `TradingComplete`.",
}


def _model_obligation(concern: str) -> str:
    return json.dumps(
        {
            "instruction": json.dumps(
                {
                    "concern": concern,
                    "concern_template": f"feature/state_model/{concern}",
                    "rules": ["model supplied"],
                    "section": "state_model",
                    "section_instruction": "model supplied",
                    "task": "model supplied",
                },
                ensure_ascii=False,
            ),
            # This mirrors the old merged graph shape: every concern carried the
            # whole section and was re-sliced only much later.
            "source_requirements": STATE_REQUIREMENTS,
        },
        ensure_ascii=False,
    )


def _obligation_payload(task: dict, concern: str) -> dict:
    for raw in task["implementation_obligations"]:
        payload = json.loads(raw)
        instruction = json.loads(str(payload.get("instruction") or "{}"))
        if instruction.get("concern") == concern:
            return payload
    raise AssertionError(concern)


def _bound_task() -> tuple[dict, list[dict]]:
    task = {"task_id": "state-regression"}
    node = {
        "symbol": "AuthoredStateModel",
        "obligations": [
            _model_obligation("variables"),
            _model_obligation("transitions"),
        ],
    }
    section, concerns = _bind_atomic_leaf_contract(task, node, STATE_REQUIREMENTS)
    assert section == "state_model"
    return task, concerns


def test_concern_provenance_is_host_sliced_before_source_generation() -> None:
    assert slice_concern_requirements(
        STATE_REQUIREMENTS, concern="variables"
    ) == {key: STATE_REQUIREMENTS[key] for key in ("R29", "R30", "R31", "R32", "R33")}

    task, _concerns = _bound_task()
    variables = _obligation_payload(task, "variables")["source_requirements"]
    transitions = _obligation_payload(task, "transitions")["source_requirements"]
    assert list(variables) == ["R29", "R30", "R31", "R32", "R33"]
    assert list(transitions) == ["R29", "R34", "R35", "R36"]


def test_markdown_variables_lower_without_model_and_never_construct_bare_enumset() -> None:
    task, concerns = _bound_task()
    variables = concerns[0]
    contracts = _state_variable_contract(task, variables)
    by_name = {item["name"]: item for item in contracts}

    assert by_name["PlayerBalance"]["java_type"] == "double"
    assert by_name["PlayerBalance"]["default_literal"] == "0.0"
    assert by_name["ShipComponents"]["java_type"] == "java.util.List<Object>"
    assert by_name["ShipComponents"]["default_literal"] == "new java.util.ArrayList<>()"
    assert by_name["PlanetControl"]["java_type"] == "java.util.Set<java.lang.Enum<?>>"
    assert by_name["PlanetControl"]["default_literal"] == "new java.util.HashSet<>()"

    members = _deterministic_state_variable_members(task, variables)
    assert "new java.util.EnumSet" not in members
    assert (
        "private static java.util.Set<java.lang.Enum<?>> PlanetControl"
        " = new java.util.HashSet<>();"
    ) in members

    def forbidden_model_call(_messages):
        raise AssertionError("variables should be host-lowered before coder decode")

    executor = AtomicConcernExecutor(
        root=Path("."),
        target=Path("AuthoredStateModel.java"),
        relative="AuthoredStateModel.java",
        symbol="AuthoredStateModel",
        original=(
            "public final class AuthoredStateModel {\n"
            "    // MMM_AUTHORED_FEATURE_BODY\n"
            "}\n"
        ),
        task=task,
        section="state_model",
        concerns=(variables,),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=forbidden_model_call,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda _path, _source: None,
    )
    result = executor.run()
    assert result["repair_count"] == 0
    assert "java.util.HashSet" in result["source"]


def test_enumset_and_collection_interface_construction_are_rejected_before_javac() -> None:
    reason, path = _semantic_component_issue(
        "fields",
        {"initializer": "new java.util.EnumSet<>()"},
    )
    assert "cannot be instantiated directly" in reason
    assert path == ["initializer"]

    reason, path = _semantic_component_issue(
        "fields",
        {"initializer": "java.util.EnumSet.noneOf(java.lang.Enum.class)"},
    )
    assert "concrete enum class" in reason
    assert path == ["initializer"]


class _Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_tool_decision(self, role, messages, **kwargs):
        assert role == "coder"
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return deepcopy(next(self.responses))


def test_semantic_repair_changes_only_bad_initializer_component() -> None:
    messages = [{
        "role": "user",
        "content": json.dumps({
            "response_region": "members",
            "host_selected_class": "AuthoredStateModel",
            "concern": {"name": "variables"},
        }),
    }]
    router = _Router([
        {
            "fields": [{
                "modifiers": ["private"],
                "type": "java.util.List",
                "name": "items",
                "initializer": "new java.util.List<>()",
            }]
        },
        {"value": "new java.util.ArrayList<>()"},
    ])

    source = _call_atomic_java_region(
        router, messages, output_token_ceiling=None
    )
    assert [kwargs["tool_name"] for _messages, kwargs in router.calls] == [
        "emit_java_structure",
        "repair_java_component",
    ]
    repair = json.loads(router.calls[1][0][-1]["content"])["component_repair"]
    assert repair["selected_path"] == ["initializer"]
    assert "new java.util.ArrayList<>()" in source
    assert "new java.util.List<>()" not in source
