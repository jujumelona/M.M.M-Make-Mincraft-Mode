from __future__ import annotations

import json
import shutil
import subprocess
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
from minecraft_mod_ai.custom_module_generator import (
    _atomic_parameters_for_request,
    _call_atomic_java_region,
)
from minecraft_mod_ai.implementation_graph_execution import _bind_atomic_leaf_contract
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

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
        STATE_REQUIREMENTS, concern="variables", require_anchor=True
    ) == {key: STATE_REQUIREMENTS[key] for key in ("R29", "R30", "R31", "R32", "R33")}
    assert slice_concern_requirements(
        {"R29": "## state_model", "R34": STATE_REQUIREMENTS["R34"]},
        concern="variables",
        require_anchor=True,
    ) == {}

    task, _concerns = _bound_task()
    variables = _obligation_payload(task, "variables")["source_requirements"]
    transitions = _obligation_payload(task, "transitions")["source_requirements"]
    assert list(variables) == ["R29", "R30", "R31", "R32", "R33"]
    assert list(transitions) == ["R29", "R34", "R35", "R36"]


def test_structured_records_are_semantic_authority_over_markdown() -> None:
    state_spec = {
        **{name: [] for name in DETAIL_RECORDS["state_model"]},
        "variables": [{
            "name": "credits",
            "owner": "Player",
            "type": "Int",
            "unit": "credits",
            "default": "0",
            "domain": "non-negative",
        }],
        "inapplicable_concerns": [
            {"concern": name, "reason": "not required"}
            for name in DETAIL_RECORDS["state_model"]
            if name != "variables"
        ],
    }
    structured = {
        "state_model": {
            "specification": state_spec,
            "constraint_evidence_refs": [],
        }
    }
    task = {"task_id": "structured-authority"}
    node = {
        "symbol": "AuthoredStateModel",
        "obligations": [_model_obligation("variables")],
    }
    section, concerns = _bind_atomic_leaf_contract(
        task,
        node,
        {
            "R12": "## state_model",
            "R13": "- variables: deliberately ambiguous human projection",
        },
        structured_sections=structured,
    )

    assert section == "state_model"
    assert [item["concern"] for item in concerns] == ["variables"]
    payload = json.loads(task["implementation_obligations"][0])
    assert payload["structured_records"] == state_spec["variables"]

    contract = _state_variable_contract(task, concerns[0])
    assert contract[0]["java_type"] == "int"
    assert contract[0]["default_literal"] == "0"


def test_structured_renderer_projects_nested_records_to_leaf_fields() -> None:
    from worksheet_fixtures import row

    from minecraft_mod_ai.authored_structured_design import render_structured_sections

    text = render_structured_sections({"behavior_contract": row("behavior_contract")})
    assert "- inputs: name type unit range default source" in text
    assert "name=behavior_contract inputs name:" in text
    assert "source=behavior_contract inputs source:" in text
    assert "identity=" not in text
    assert "constraints=" not in text


def test_graph_frontend_assigns_each_state_page_to_only_its_actual_concern() -> None:
    from minecraft_mod_ai.implementation_decisions import compile_contribution

    def compile_page(requirements):
        return compile_contribution(
            object(),
            "compile_implementation_graph",
            {
                "requirements": requirements,
                "unit_context": {"R12": "## state_model", **requirements},
                "accepted_nodes": [],
                "planned_units": ["state_model"],
                "current_units": ["state_model"],
                "unit_ids": ["state_model"],
                "unresolved_dependencies": [],
                "platform": {},
                "project_context": "",
                "package": "example",
                "mod_id": "test",
                "page": 1,
            },
            {},
            lambda: None,
        )["nodes"][0]

    variables = compile_page({
        "R13": (
            "- variables: 현재 돈 (Int, owner=Player), "
            "보유 재료 (Map<String, Int>, owner=Inventory)"
        )
    })
    transitions = compile_page({
        "R14": (
            "- transitions: from_state(준비됨) -> trigger(재료 구매/제작) "
            "-> to_state(조립중)"
        )
    })

    assert len(variables["obligations"]) == 1
    assert len(transitions["obligations"]) == 1

    variable_payload = json.loads(variables["obligations"][0])
    transition_payload = json.loads(transitions["obligations"][0])
    variable_instruction = json.loads(variable_payload["instruction"])
    transition_instruction = json.loads(transition_payload["instruction"])

    assert variable_instruction["concern"] == "variables"
    assert list(variable_payload["source_requirements"]) == ["R12", "R13"]
    assert transition_instruction["concern"] == "transitions"
    assert list(transition_payload["source_requirements"]) == ["R12", "R14"]


def test_full_state_unit_context_does_not_duplicate_sibling_concerns_per_page() -> None:
    from minecraft_mod_ai.implementation_decisions import compile_contribution

    full_state = {
        "R12": "## state_model",
        "R13": "- variables: credits (Int, owner=Player)",
        "R14": "- transitions: from_state(idle) -> trigger(buy) -> to_state(done)",
        "R15": "- invariants: credits >= 0",
        "R16": "- initialization: server start resets temporary state",
        "R17": "- updates: trade completion increments credits",
        "R18": "- cleanup: player death releases temporary ownership",
        "R19": "- concurrency: one owner controls each ship",
    }
    common = {
        "unit_context": full_state,
        "accepted_nodes": [],
        "planned_units": ["state_model"],
        "current_units": ["state_model"],
        "unit_ids": ["state_model"],
        "unresolved_dependencies": [],
        "platform": {},
        "project_context": "",
        "package": "example",
        "mod_id": "test",
        "page": 1,
    }

    page = compile_contribution(
        object(),
        "compile_implementation_graph",
        {"requirements": {"R14": full_state["R14"]}, **common},
        {},
        lambda: None,
    )["nodes"][0]

    assert len(page["obligations"]) == 1
    payload = json.loads(page["obligations"][0])
    instruction = json.loads(payload["instruction"])
    assert instruction["concern"] == "transitions"
    assert list(payload["source_requirements"]) == ["R12", "R14"]


def test_leaf_contract_materializes_only_concerns_present_in_authored_source() -> None:
    requirements = {
        "R12": "## state_model",
        "R13": "- variables: credits (Int, owner=Player)",
        "R14": "- transitions: from_state(idle) -> trigger(buy) -> to_state(done)",
    }
    task = {"task_id": "exact-active-concerns"}
    node = {
        "symbol": "AuthoredStateModel",
        "obligations": [
            _model_obligation("variables"),
            _model_obligation("transitions"),
            _model_obligation("invariants"),
            _model_obligation("initialization"),
            _model_obligation("updates"),
            _model_obligation("cleanup"),
            _model_obligation("concurrency"),
        ],
    }
    section, active = _bind_atomic_leaf_contract(task, node, requirements)
    assert section == "state_model"
    assert [item["concern"] for item in active] == ["variables", "transitions"]
    assert len(task["implementation_obligations"]) == 2
    payloads = [json.loads(raw) for raw in task["implementation_obligations"]]
    assert [json.loads(p["instruction"])["concern"] for p in payloads] == [
        "variables",
        "transitions",
    ]
    assert list(payloads[0]["source_requirements"]) == ["R12", "R13"]
    assert list(payloads[1]["source_requirements"]) == ["R12", "R14"]


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


def test_inline_compact_state_variables_are_host_lowered_without_coder(tmp_path) -> None:
    requirements = {
        "R12": "## state_model",
        "R13": (
            "- variables: 현재 돈 (Int, owner=Player), "
            "보유 재료 (Map<String, Int>, owner=Inventory), "
            "우주선 구성 (List<ShipPartConfig>, owner=WorldData), "
            "위치 좌표 (Vec3Double, owner=Server)"
        ),
        "R14": (
            "- transitions: from_state(준비됨) -> trigger(재료 구매/제작) "
            "-> to_state(조립중)"
        ),
    }
    task = {"task_id": "inline-state-regression"}
    node = {"symbol": "AuthoredStateModel", "obligations": [_model_obligation("variables")]}
    _section, concerns = _bind_atomic_leaf_contract(task, node, requirements)
    variables = concerns[0]
    contracts = _state_variable_contract(task, variables)
    assert [item["java_type"] for item in contracts] == [
        "int",
        "java.util.Map<String, Integer>",
        "java.util.List<ShipPartConfig>",
        "Vec3Double",
    ]

    members = _deterministic_state_variable_members(task, variables)
    assert "private static final class ShipPartConfig {}" in members
    assert "private static final class Vec3Double {}" in members
    assert "new java.util.HashMap<>()" in members
    assert "new java.util.ArrayList<>()" in members

    def forbidden_model_call(_messages):
        raise AssertionError("inline variables must be host-lowered before coder decode")

    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / "AuthoredStateModel.java",
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
    source = executor.run()["source"]
    assert "uD604uC7AC_uB3C8" in source
    assert "<init>" not in source

    javac = shutil.which("javac")
    if javac:
        target = tmp_path / "AuthoredStateModel.java"
        target.write_text(source, encoding="utf-8")
        compiled = subprocess.run([javac, str(target)], capture_output=True, text=True, check=False)
        assert compiled.returncode == 0, compiled.stderr


def test_structured_state_model_is_host_compiled_in_one_pass_without_coder(tmp_path) -> None:
    specification = {
        **{name: [] for name in DETAIL_RECORDS["state_model"]},
        "variables": [{
            "name": "credits",
            "owner": "Player",
            "type": "Int",
            "unit": "credits",
            "default": "100",
            "domain": "non-negative",
        }],
        "transitions": [{
            "from_state": "idle",
            "trigger": "buy",
            "guard": "credits >= cost",
            "mutation": "credits -= cost",
            "to_state": "done",
        }],
        "invariants": [{
            "condition": "credits >= 0",
            "enforcement": "reject negative balance",
        }],
        "initialization": [{
            "owner": "Player",
            "trigger": "server_start",
            "initial_state": "credits = 100",
        }],
        "updates": [{
            "trigger": "reward",
            "mutation": "credits += amount",
            "owner": "Player",
        }],
        "cleanup": [{
            "event": "reset",
            "action": "credits = 0",
            "retained_state": "none",
        }],
        "concurrency": [{
            "entry_path": "state_runtime",
            "ownership": "server",
            "reentrancy_rule": "serialized",
        }],
        "inapplicable_concerns": [],
    }
    structured = {
        "state_model": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }
    requirements = {
        "R1": "## state_model",
        "R2": "- variables: name owner type unit default domain",
        "R3": "- transitions: from_state trigger guard mutation to_state",
        "R4": "- invariants: condition enforcement",
        "R5": "- initialization: owner trigger initial_state",
        "R6": "- updates: trigger mutation owner",
        "R7": "- cleanup: event action retained_state",
        "R8": "- concurrency: entry_path ownership reentrancy_rule",
    }
    task = {"task_id": "structured-state-runtime"}
    node = {
        "symbol": "AuthoredStateModel",
        "obligations": [
            _model_obligation(name)
            for name in DETAIL_RECORDS["state_model"]
        ],
    }
    section, active = _bind_atomic_leaf_contract(
        task,
        node,
        requirements,
        structured_sections=structured,
    )
    assert section == "state_model"
    assert [row["concern"] for row in active] == list(
        DETAIL_RECORDS["state_model"]
    )

    def forbidden_model_call(_messages):
        raise AssertionError("structured state_model must never call the coder")

    writes = {}
    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java",
        symbol="AuthoredStateModel",
        original=(
            "public final class AuthoredStateModel {\n"
            "    // MMM_AUTHORED_FEATURE_BODY\n"
            "}\n"
        ),
        task=task,
        section="state_model",
        concerns=tuple(active),
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=forbidden_model_call,
        compile_java=lambda _root: SimpleNamespace(status="PASS"),
        compile_log=lambda _report: "",
        write_source=lambda path, source: writes.__setitem__(str(path), source),
    )
    result = executor.run()
    assert result["repair_count"] == 0
    source = result["source"]
    assert "$mmmTransitions" in source
    assert "context ->" in source
    assert "public static synchronized String transition" in source

    javac = shutil.which("javac")
    if javac:
        target = tmp_path / "AuthoredStateModel.java"
        target.write_text(source, encoding="utf-8")
        compiled = subprocess.run(
            [javac, str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr


def test_top_level_method_schema_structurally_forbids_outer_constructor() -> None:
    parameters, _shape = _atomic_parameters_for_request(
        {"host_selected_class": "AuthoredStateModel", "concern": {"name": "variables"}},
        response_region="members",
    )
    top_level_method = parameters["properties"]["methods"]["items"]
    nested_method = parameters["properties"]["classes"]["items"]["properties"]["methods"]["items"]
    assert "<init>" not in top_level_method["properties"]["name"]["pattern"]
    assert "<init>" in nested_method["properties"]["name"]["pattern"]


def test_explicit_collection_element_types_are_preserved() -> None:
    from minecraft_mod_ai.atomic_concern_source import _state_java_contract

    assert _state_java_contract("List<ShipPart>", "[]") == (
        "java.util.List<ShipPart>",
        "new java.util.ArrayList<>()",
    )
    assert _state_java_contract("EnumSet<ColonyState>", "empty_set") == (
        "java.util.EnumSet<ColonyState>",
        "java.util.EnumSet.noneOf(ColonyState.class)",
    )
    assert _state_java_contract("EnumSet<?>", "empty_set") == ("", "")


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


def test_invalid_collection_initializer_is_terminal_without_repair() -> None:
    import pytest

    from minecraft_mod_ai.custom_module_errors import AtomicJavaDecisionError

    messages = [{"role": "user", "content": json.dumps({
        "response_region": "members", "host_selected_class": "AuthoredStateModel",
        "concern": {"name": "variables"},
    })}]
    router = _Router([
        {"part": "fields"},
        {"type": "java.util.List", "name": "items", "initializer": "new java.util.List<>()"},
        {"part": "done"},
    ])
    with pytest.raises(AtomicJavaDecisionError, match="cannot be instantiated directly"):
        _call_atomic_java_region(router, messages, output_token_ceiling=None)
    assert len(router.calls) == 3
    assert all(kwargs["tool_name"] == "emit_java_part" for _, kwargs in router.calls)
