from copy import deepcopy
from typing import Any

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir


class NativePlanner:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
        assert role == "planner"
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == {"value"}
        self.calls.append((tool_name, messages, parameters))
        return {"value": next(self.answers)}

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        self.calls.append((role, messages, kwargs))
        import json
        ans = next(self.answers)
        if isinstance(ans, dict) and ans.get("op") == "literal" and ans.get("type") in ("int", "long", "double"):
            ans = dict(ans)
            ans["value"] = str(ans["value"])
        return json.dumps({"value": ans})


def test_native_recursive_operation_builder_preserves_nested_arithmetic():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    router = NativePlanner([
        {"op": "let", "name": "credits", "type": "int"},
        {"op": "binary", "operator": "+"},
        {"op": "literal", "type": "int", "value": 7},
        {"op": "literal", "type": "int", "value": 3},
        {"op": "assert", "message": "credits must total ten"},
        {"op": "binary", "operator": "=="},
        {"op": "ref", "name": "credits"},
        {"op": "literal", "type": "int", "value": 10},
        {"op": "done"},
    ])
    author = TypedOperationAuthor(router, "Credit test", {}, {}, max_calls=40)
    body = author.body("requirement_0001")
    assert body[0] == {
        "op": "let", "name": "credits", "type": "int",
        "value": {"op": "binary", "operator": "+",
                  "left": {"op": "literal", "type": "int", "value": 7},
                  "right": {"op": "literal", "type": "int", "value": 3}},
    }
    assert body[1]["condition"]["left"] == {"op": "ref", "name": "credits"}
    assert len(router.calls) == 9


def test_host_resolves_fully_determined_schema_without_model_call():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    router = NativePlanner([])
    author = TypedOperationAuthor(router, "fixed schema", {}, {}, max_calls=1)
    value = author._ask(
        "deterministic_fixture",
        {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["host"]},
                "count": {"type": "integer", "const": 2},
                "tags": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string", "const": "fixed"},
                },
            },
            "required": ["mode", "count", "tags"],
            "additionalProperties": False,
        },
        scope="fixture",
    )

    assert value == {
        "mode": "host",
        "count": 2,
        "tags": ["fixed", "fixed"],
    }
    assert author.call_count == 0
    assert router.calls == []


@pytest.mark.parametrize("combinator", ["oneOf", "anyOf", "allOf"])
def test_host_resolves_single_schema_branch_without_model_call(combinator):
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    router = NativePlanner([])
    author = TypedOperationAuthor(router, "single branch", {}, {}, max_calls=1)
    value = author._ask(
        "single_branch_fixture",
        {
            combinator: [
                {
                    "type": "object",
                    "properties": {"mode": {"const": "host"}},
                    "required": ["mode"],
                    "additionalProperties": False,
                }
            ],
            "description": "There is no semantic choice for the model to make.",
        },
        scope="fixture",
    )

    assert value == {"mode": "host"}
    assert author.call_count == 0
    assert router.calls == []


def test_native_recursive_authoring_stops_at_hard_call_bound():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor

    author = TypedOperationAuthor(
        NativePlanner([
            {"op": "unary", "operator": "!"},
            {"op": "unary", "operator": "!"},
            {"op": "unary", "operator": "!"},
        ]),
        "test",
        {},
        {},
        max_calls=3,
    )
    with pytest.raises(ValueError, match="TYPED_PLAN_AUTHORING_LIMIT"):
        author.expression("condition")


def test_legacy_saved_plan_roundtrip_remains_readable():
    legacy = AuthoredPlan.from_dict({
        "schema_version": "mmm/authored-plan-v1", "requested_prompt": "test", "text": "test",
    })
    assert legacy.typed_plan_ir == {}
    restored = AuthoredPlan.from_dict(deepcopy(legacy.to_dict()))
    assert restored.calculate_hash() == legacy.calculate_hash()

def test_event_handler_signature_is_host_owned_during_authoring():
    specification = {
        name: []
        for name in DETAIL_RECORDS["integration"]
    }
    specification["entry_points"] = [
        {
            "boundary": "server",
            "trigger": "server startup",
            "owner": "server",
        }
    ]
    specification["inapplicable_concerns"] = []
    structured = {
        "integration": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }
    router = NativePlanner([])

    plan = author_typed_plan_ir(
        router,
        "server event design",
        structured,
        {},
        max_calls=16,
    )

    assert plan["event_bindings"] == [
        {
            "event": "server_started",
            "function": "entryPoint1_server_started",
            "entry_point_index": 0,
            "config": {},
        }
    ]
    handler = plan["functions"][0]
    assert handler["id"] == "entryPoint1_server_started"
    assert handler["parameters"] == [
        {"name": "server", "type": "object"}
    ]
    assert handler["return_type"] == "void"
    assert handler["body"] == []
    assert router.calls == []
    requested_fields = [
        call[1][1]["content"]
        for call in router.calls
    ]
    assert all("parameter_name" not in payload for payload in requested_fields)
    assert all("return_type" not in payload for payload in requested_fields)


def test_invalid_literal_int_string_fails_closed_safely():
    from minecraft_mod_ai.typed_plan_authoring import _decode_int_literal

    with pytest.raises(ValueError, match="TYPED_PLAN_LITERAL_INT_INVALID"):
        _decode_int_literal("}}}null}}}}", scope="test")

    with pytest.raises(ValueError, match="TYPED_PLAN_LITERAL_INT_RANGE"):
        _decode_int_literal("99999999999", scope="test")


def test_semantic_game_dispatch_lowers_deterministically():
    from minecraft_mod_ai.typed_plan_authoring import lower_semantic_game_dispatch_to_ir
    from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir

    rules = [
        {
            "trigger_event": "player_join",
            "state_key": "credits",
            "action_kind": "increment_state",
            "value": 7,
        },
        {
            "trigger_event": "any",
            "state_key": "credits",
            "action_kind": "set_state",
            "value": 10,
        },
    ]
    statements = lower_semantic_game_dispatch_to_ir(rules, {"credits": "int"}, {})
    assert len(statements) == 3
    assert statements[0]["op"] == "if"
    assert statements[0]["condition"]["right"]["value"] == "player_join"
    assert statements[1]["op"] == "state_set"
    assert statements[2]["op"] == "return"

    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "0" * 64,
        "functions": [
            {
                "id": "logic_dispatch",
                "parameters": [
                    {"name": "event", "type": "string"},
                    {"name": "primary", "type": "object"},
                    {"name": "secondary", "type": "object"},
                    {"name": "flag", "type": "boolean"},
                ],
                "return_type": "int",
                "body": statements,
                "covers": ["behavior_contract.rules"],
            }
        ],
        "initialize": [],
        "platform_modules": [],
        "event_bindings": [],
    }
    validated = validate_typed_plan_ir(plan)
    assert validated["functions"][0]["id"] == "logic_dispatch"


@pytest.mark.parametrize("value", ["shipyard", "SHIPYARD", " shipyard ", "string"])
def test_string_state_literals_are_real_writes_even_when_they_match_names(value):
    from minecraft_mod_ai.typed_plan_authoring import lower_semantic_game_dispatch_to_ir

    body = lower_semantic_game_dispatch_to_ir([{
        "trigger_event": "command", "action_kind": "set_state",
        "state_key": "shipyard", "value": value,
    }], {"shipyard": "string"}, {})
    write = body[0]["then"][0]
    assert write["op"] == "state_set"
    assert write["key"]["value"] == "shipyard"
    assert write["value"] == {"op": "literal", "type": "string", "value": value}


def test_semantic_dispatch_accepts_literal_state_write_without_reauthoring(monkeypatch):
    from minecraft_mod_ai import typed_plan_authoring, authored_structured_design

    def concerns(_sections, section):
        if section == "state_model":
            return {"variables": [{"name": "resources", "type": "string"}]}
        if section == "resources_and_ui":
            return {"interactions": [{
                "server_action": "change_status", "trigger": "button",
            }]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    attempts = []
    def fake_generate(*_args, **_kwargs):
        attempts.append(1)
        return {"trigger_event": "command", "action_kind": "set_state",
                "state_key": "resources", "value": "resources"}

    monkeypatch.setattr(
        typed_plan_authoring, "generate_fixed_template_value", fake_generate,
    )
    rules = typed_plan_authoring.author_semantic_game_dispatch(
        None, "make a working purchase", {}, {},
        coverage_refs=("resources_and_ui.interactions",),
        bound_events=("command",),
    )
    assert len(attempts) == 1
    assert rules[0]["then"][0]["value"]["value"] == "resources"


def test_atomic_string_mutation_lowers_without_discarding_state(monkeypatch):
    from minecraft_mod_ai import typed_plan_authoring, authored_structured_design

    def concerns(_sections, section):
        if section == "state_model":
            return {"variables": [{"name": "shipyard", "type": "string"}]}
        if section == "algorithm":
            return {"atomic_mutations": [{"mutations": "update shipyard state"}]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    calls = []

    def model(*_args, **kwargs):
        calls.append(kwargs["response_schema"])
        return {"trigger_event": "command", "action_kind": "set_state",
                "state_key": "shipyard", "value": "shipyard"}

    monkeypatch.setattr(typed_plan_authoring, "generate_fixed_template_value", model)
    body = typed_plan_authoring.author_semantic_game_dispatch(
        None, "update shipyard state", {}, {},
        coverage_refs=("algorithm.atomic_mutations",), bound_events=("command",),
    )
    assert len(calls) == 1
    assert body[0]["then"][0]["key"]["value"] == "shipyard"
    assert body[0]["then"][0]["value"]["value"] == "shipyard"


def test_planner_budget_enforces_limit_and_tracks_stages():
    from minecraft_mod_ai.planner_budget import PlannerBudget

    budget = PlannerBudget(max_calls=3)
    budget.consume("structured.state", 2)
    assert budget.call_count == 2
    assert budget.remaining() == 1
    assert budget.stage_counts["structured.state"] == 2

    budget.consume("typed.dispatch", 1)
    assert budget.remaining() == 0

    with pytest.raises(ValueError, match="PLANNER_BUDGET_EXCEEDED"):
        budget.consume("typed.dispatch", 1)


def test_all_platform_authoring_schemas_satisfy_atomic_ceiling():
    from minecraft_mod_ai.model_output_atomicity_contract import structured_output_token_ceiling
    from minecraft_mod_ai.typed_plan_authoring import (
        TypedOperationAuthor,
        _author_platform_config,
    )
    from minecraft_mod_ai.typed_platform_ir import (
        PLATFORM_HOST_KINDS,
        PLATFORM_KINDS,
        validate_platform_modules,
    )

    def _make_dummy_val(s):
        if not isinstance(s, dict):
            return {}
        if "oneOf" in s and s["oneOf"]:
            return _make_dummy_val(s["oneOf"][0])
        if "anyOf" in s and s["anyOf"]:
            return _make_dummy_val(s["anyOf"][0])
        if "const" in s:
            return s["const"]
        if "enum" in s and s["enum"]:
            return s["enum"][0]
        t = s.get("type")
        if t == "string":
            pat = s.get("pattern", "")
            if "#" in pat:
                return "#FF0000"
            if pat.startswith("^-") or ("0-9" in pat and "a-z" not in pat):
                return "1"
            if ":[a-z" in pat or "minecraft:" in pat:
                return "minecraft:stone"
            if "a-z" in pat:
                return "test"
            return "mock_name"
        if t == "integer":
            return int(s.get("minimum", 1))
        if t == "number":
            return float(s.get("minimum", 1.0))
        if t == "boolean":
            return True
        if t == "object" or "properties" in s:
            res = {}
            for req in s.get("required", ()):
                res[req] = _make_dummy_val(s.get("properties", {}).get(req, {}))
            return res
        return {}

    class AtomicBudgetTrackingRouter:
        def __init__(self):
            self.queried_schemas = []

        def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
            raise NotImplementedError()

        def generate_text(self, role, messages, *, response_schema=None, **kwargs):
            import json
            if response_schema:
                self.queried_schemas.append(response_schema)
                structured_output_token_ceiling(response_schema, absolute_ceiling=4096)
                val_schema = response_schema.get("properties", {}).get("value", {})
                return json.dumps({"value": _make_dummy_val(val_schema)})
            return json.dumps({"value": {}})

    for kind in sorted(PLATFORM_KINDS):
        router = AtomicBudgetTrackingRouter()
        author = TypedOperationAuthor(router, "test prompt", {}, {}, max_calls=10)
        cfg = _author_platform_config(
            author,
            kind,
            scope="test_scope",
            structured_sections={},
            module_id=f"mod_{kind}",
            uncovered=set(),
        )
        if kind in {"recipe", "advancement", "loot", "tag", "command"}:
            covers = ["resources_and_ui.paths"]
        elif kind == "networking":
            covers = ["resources_and_ui.interactions"]
        elif kind == "network_sync":
            covers = ["authority_and_network.synchronization"]
        elif kind == "state_store":
            covers = ["persistence.stored_state"]
        elif kind == "resource_policy":
            covers = ["resources_and_ui.accessibility"]
        else:
            covers = ["resources_and_ui.registries"]

        modules = [{"module_id": f"mod_{kind}", "kind": kind, "config": cfg, "covers": covers}]
        if kind == "skill":
            modules.insert(0, {
                "module_id": "warrior",
                "kind": "class",
                "config": {"display_name": "Warrior"},
                "covers": ["resources_and_ui.registries"],
            })
        validated = validate_platform_modules(modules)
        assert len(validated) >= 1

        if kind in PLATFORM_HOST_KINDS or kind in {"recipe", "advancement", "loot"}:
            assert author.call_count == 0, f"{kind} must be resolved deterministically by host"
        elif kind in {"networking", "gui", "tag"}:
            assert author.call_count == 2, f"{kind} decomposes into 2 atomic model decisions"
        else:
            assert author.call_count == 1, f"{kind} should ask the model exactly once"
            for schema in router.queried_schemas:
                ceiling = structured_output_token_ceiling(schema, absolute_ceiling=4096)
                assert ceiling <= 4096, f"{kind} schema ceiling {ceiling} exceeds 4096"


def test_assert_model_atomic_decision_schema_enforces_boundaries():
    from minecraft_mod_ai.typed_plan_authoring import (
        assert_model_atomic_decision_schema,
        _SHOP_ENTRY_SCHEMA,
        _ENTITY_TRAITS_AUTHOR_SCHEMA,
        _QUEST_PARAMS_AUTHOR_SCHEMA,
        _SKILL_PARAMS_AUTHOR_SCHEMA,
    )
    from minecraft_mod_ai.typed_platform_ir import platform_config_schema

    # Valid leaf schemas must pass without error
    assert_model_atomic_decision_schema(_SHOP_ENTRY_SCHEMA, field="shop_entry")
    assert_model_atomic_decision_schema(_ENTITY_TRAITS_AUTHOR_SCHEMA, field="entity_traits")
    assert_model_atomic_decision_schema(_QUEST_PARAMS_AUTHOR_SCHEMA, field="quest_params")
    assert_model_atomic_decision_schema(_SKILL_PARAMS_AUTHOR_SCHEMA, field="skill_params")
    assert_model_atomic_decision_schema({"type": "string", "enum": ["a", "b"]}, field="choice")

    # Asking for "platform_config" directly is forbidden
    with pytest.raises(ValueError, match="platform_config"):
        assert_model_atomic_decision_schema({"type": "string"}, field="platform_config")

    # Arrays are forbidden
    with pytest.raises(ValueError, match="array"):
        assert_model_atomic_decision_schema(
            {"type": "array", "items": {"type": "string"}}, field="arr"
        )

    # Property containing an array is forbidden
    with pytest.raises(ValueError, match="array"):
        assert_model_atomic_decision_schema(
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "additionalProperties": False,
            },
            field="obj_with_arr",
        )

    # Nested objects are forbidden
    with pytest.raises(ValueError, match="nested object"):
        assert_model_atomic_decision_schema(
            {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "object",
                        "properties": {"k": {"type": "string"}},
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            },
            field="nested_obj",
        )

    # Canonical storage schemas are forbidden
    with pytest.raises(ValueError, match="canonical storage schema"):
        assert_model_atomic_decision_schema(platform_config_schema("item"), field="item")

    with pytest.raises(ValueError, match="canonical storage schema|array"):
        assert_model_atomic_decision_schema(platform_config_schema("quest"), field="quest")

    # additionalProperties: False is required on objects
    with pytest.raises(ValueError, match="additionalProperties: False"):
        assert_model_atomic_decision_schema(
            {"type": "object", "properties": {"a": {"type": "string"}}},
            field="open_obj",
        )

    # Limit on number of properties
    with pytest.raises(ValueError, match="limit of 12"):
        assert_model_atomic_decision_schema(
            {
                "type": "object",
                "properties": {f"k_{i}": {"type": "string"} for i in range(13)},
                "additionalProperties": False,
            },
            field="too_many_props",
        )


def test_author_typed_plan_ir_shop_platform_module_within_budget():
    specification = {
        name: []
        for name in DETAIL_RECORDS["resources_and_ui"]
    }
    specification["registries"] = [
        {
            "purpose": "shop",
            "identifier": "shop_keeper",
            "binding_requirement": "register one shop keeper",
        }
    ]
    specification["inapplicable_concerns"] = []
    structured = {
        "resources_and_ui": {
            "specification": specification,
            "constraint_evidence_refs": [],
        }
    }

    class ShopPlanner:
        def __init__(self):
            self.calls = []

        def generate_tool_decision(self, role, messages, *, tool_name, parameters, **kwargs):
            raise NotImplementedError()

        def generate_text(self, role, messages, *, response_schema=None, **kwargs):
            import json
            from minecraft_mod_ai.model_output_atomicity_contract import structured_output_token_ceiling

            if response_schema:
                structured_output_token_ceiling(response_schema, absolute_ceiling=4096)

            self.calls.append((messages, response_schema))
            if len(self.calls) == 1:
                return json.dumps({"value": "shop"})
            return json.dumps({
                "value": {
                    "item": "minecraft:diamond_sword",
                    "count": 1,
                    "price": 150.0,
                }
            })

    router = ShopPlanner()
    plan = author_typed_plan_ir(
        router,
        "create a shop with custom items",
        structured,
        {},
        max_calls=16,
    )
    assert len(plan["platform_modules"]) == 1
    shop_module = plan["platform_modules"][0]
    assert shop_module["kind"] == "shop"
    assert shop_module["config"]["entries"] == [
        {
            "id": "typed_shop_1_entry_1",
            "item": "minecraft:diamond_sword",
            "count": 1,
            "price": 150.0,
        }
    ]


def test_author_typed_plan_ir_filters_by_deterministic_module_kinds() -> None:
    res_spec = {name: [] for name in DETAIL_RECORDS["resources_and_ui"]}
    res_spec["registries"] = [
        {
            "purpose": "item",
            "identifier": "ruby",
            "binding_requirement": "ruby item",
        }
    ]
    res_spec["inapplicable_concerns"] = []

    int_spec = {name: [] for name in DETAIL_RECORDS["integration"]}
    int_spec["entry_points"] = [
        {
            "boundary": "server",
            "trigger": "/give_ruby",
            "owner": "server",
        }
    ]
    int_spec["inapplicable_concerns"] = []

    structured = {
        "resources_and_ui": {
            "specification": res_spec,
            "constraint_evidence_refs": [],
        },
        "integration": {
            "specification": int_spec,
            "constraint_evidence_refs": [],
        },
    }

    class TestRouter:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any]] = []

        def generate_text(self, role: Any, messages: Any, *, response_schema: Any = None, **kwargs: Any) -> str:
            import json

            self.calls.append((messages, response_schema))
            val_schema = response_schema.get("properties", {}).get("value", response_schema)
            if "enum" in val_schema:
                # Ensure command was excluded by allowed_platform_kinds
                assert "command" not in val_schema["enum"]
                assert "item" in val_schema["enum"]
                return json.dumps({"value": "item"})
            if val_schema.get("type") == "object":
                res = {}
                for k in val_schema.get("properties", {}):
                    res[k] = "Ruby"
                return json.dumps({"value": res})
            return json.dumps({"value": "Ruby"})

    router = TestRouter()
    plan = author_typed_plan_ir(
        router,
        "ruby item with command",
        structured,
        {},
        deterministic_module_kinds={"item", "block", "recipe", "loot", "tag"},
        max_calls=16,
    )
    assert len(plan["platform_modules"]) == 1
    assert plan["platform_modules"][0]["kind"] == "item"
    assert any(b.get("event") == "command" for b in plan["event_bindings"])


def test_complete_planner_does_not_compile_execution_at_plan_time(monkeypatch: Any) -> None:
    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope", lambda *a, **k: {})
    from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
    from minecraft_mod_ai.model_router import ModelRouter

    received_kinds = None

    def fake_author_typed_plan_ir(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal received_kinds
        received_kinds = kwargs.get("deterministic_module_kinds")
        return {
            "source_sha256": "fake",
            "functions": [],
            "initialize": [],
            "platform_modules": [],
            "event_bindings": [],
        }

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.author_typed_plan_ir",
        fake_author_typed_plan_ir,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.authored_structured_design.author_structured_sections",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.authored_structured_design.render_structured_sections",
        lambda *args: "mock text",
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_support.assert_typed_plan_host_support",
        lambda *args, **kwargs: None,
    )

    class DummyAdapter:
        deterministic_module_kinds = frozenset({"item", "block"})

    router = ModelRouter(profile="fast_test")
    planner = CompleteGameDesignPlanner(router, adapter=DummyAdapter())
    planned = planner.plan("test prompt")

    assert planned.typed_plan_ir == {}
    assert received_kinds is None

def test_complete_planner_unbound_auto_defers_ir_to_production(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope", lambda *a, **k: {})
    from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
    from minecraft_mod_ai.model_router import ModelRouter

    received_allowed = "unset"
    received_kinds = "unset"

    def fake_author_typed_plan_ir(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal received_allowed, received_kinds
        received_allowed = kwargs.get("allowed_platform_kinds")
        received_kinds = kwargs.get("deterministic_module_kinds")
        return {
            "source_sha256": "fake",
            "functions": [],
            "initialize": [],
            "platform_modules": [],
            "event_bindings": [],
        }

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.author_typed_plan_ir",
        fake_author_typed_plan_ir,
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.authored_structured_design.author_structured_sections",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.authored_structured_design.render_structured_sections",
        lambda *args: "mock text",
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_support.assert_typed_plan_host_support",
        lambda *args, **kwargs: None,
    )

    router = ModelRouter(profile="fast_test")
    planner = CompleteGameDesignPlanner(router)
    planned = planner.plan("test prompt")

    assert planned.typed_plan_ir == {}
    assert received_kinds == "unset"
    assert received_allowed == "unset"


def test_permission_capability_schema_exposes_host_range_to_model():
    from minecraft_mod_ai.model_output_atomicity_contract import (
        effective_model_transport_schema,
    )
    from minecraft_mod_ai.typed_host_capabilities import (
        typed_host_capability_contracts,
    )
    from minecraft_mod_ai.typed_plan_authoring import semantic_dispatch_schema

    schema = semantic_dispatch_schema(
        {},
        typed_host_capability_contracts(),
    )
    permission = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"].get("capability_id", {}).get("const")
        == "command.has_permission"
    )
    logical = permission["properties"]["arg_1"]
    assert logical == {
        "type": "integer",
        "minimum": 0,
        "maximum": 4,
    }

    transport = effective_model_transport_schema(permission)
    assert transport["properties"]["arg_1"]["enum"] == [0, 1, 2, 3, 4]


def test_permission_capability_lowering_rejects_semantically_invalid_level():
    from minecraft_mod_ai.typed_host_capabilities import (
        typed_host_capability_contracts,
    )
    from minecraft_mod_ai.typed_plan_authoring import (
        lower_semantic_game_dispatch_to_ir,
    )

    with pytest.raises(
        ValueError,
        match="TYPED_PLAN_SEMANTIC_CAPABILITY_ARG_RANGE",
    ):
        lower_semantic_game_dispatch_to_ir(
            [
                {
                    "trigger_event": "command",
                    "action_kind": "call_capability",
                    "capability_id": "command.has_permission",
                    "arg_1": 5,
                }
            ],
            {},
            typed_host_capability_contracts(),
        )



def test_mutating_ui_commands_are_individually_executable(monkeypatch):
    """Host binds separate commands and forwards each exact authored action."""
    from minecraft_mod_ai import typed_plan_authoring, authored_structured_design
    from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues

    actions = ("ship_part_purchase", "ship_part_upgrade")

    def concerns(_sections, section):
        if section == "algorithm":
            return {"atomic_mutations": [{"mutations": "spend_and_apply"}]}
        if section == "resources_and_ui":
            return {"interactions": [
                {"server_action": action, "trigger": "player_click"}
                for action in actions
            ]}
        if section == "integration":
            return {"entry_points": [{
                "trigger": "command:shipyard", "boundary": "server", "owner": "ship"
            }]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    monkeypatch.setattr(typed_plan_authoring, "_active_concern_refs", lambda _s: (
        "algorithm.atomic_mutations", "resources_and_ui.interactions",
        "integration.entry_points",
    ))
    monkeypatch.setattr(typed_plan_authoring, "_integration_entry_points",
                        lambda _s: tuple(concerns({}, "integration")["entry_points"]))
    def semantic(_router, _source, _sections, _caps, **_kwargs):
        statements = []
        for action in actions:
            statements.append({
                "op": "if",
                "condition": {
                    "op": "binary", "operator": "==",
                    "left": {"op": "ref", "name": "event"},
                    "right": {"op": "literal", "type": "string", "value": action},
                },
                "then": [{
                    "op": "state_set",
                    "key": {"op": "literal", "type": "string", "value": "credits"},
                    "value": {"op": "literal", "type": "int", "value": 1},
                    "context": {"op": "map", "entries": []},
                }],
                "else": [],
            })
        statements.append({"op": "return", "value": {
            "op": "literal", "type": "int", "value": 1,
        }})
        return statements

    monkeypatch.setattr(typed_plan_authoring, "author_semantic_game_dispatch", semantic)
    plan = author_typed_plan_ir(None, "purchase and upgrade ships", {}, {})
    bindings = [b for b in plan["event_bindings"] if b["event"] == "command"]
    assert len(bindings) == 3
    assert [b["config"]["literal"] for b in bindings] == [
        "shipyard", "mmm_action_1", "mmm_action_2",
    ]
    functions = {f["id"]: f for f in plan["functions"]}
    for ordinal, action in enumerate(actions, start=1):
        fn = functions[f"uiAction{ordinal}_command"]
        assert fn["body"][0]["value"]["args"][0]["value"] == action

    # Check strict typed support, including synthetic action entry point indexes.
    from minecraft_mod_ai import typed_plan_support
    monkeypatch.setattr(typed_plan_support, "normalize_structured_sections", lambda s: s)
    monkeypatch.setattr(typed_plan_support, "active_concern_records", concerns)
    assert typed_plan_support_issues({}, plan) == ()


def test_ui_mutation_action_guard_uses_exact_action_and_java_value_equality():
    from minecraft_mod_ai.typed_plan_authoring import lower_semantic_game_dispatch_to_ir
    from minecraft_mod_ai.typed_plan_java import render_typed_plan_java

    action = "market_exchange_service exposes ship_part_purchase and ship_part"
    body = lower_semantic_game_dispatch_to_ir([{
        "trigger_event": "command",
        "action_guard": action,
        "action_kind": "set_state",
        "state_key": "credits",
        "value": 3,
    }], {"credits": "int"}, {})
    assert body[0]["condition"]["right"]["value"] == action
    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "0" * 64,
        "functions": [{
            "id": "logic_dispatch",
            "parameters": [
                {"name": "event", "type": "string"},
                {"name": "primary", "type": "object"},
                {"name": "secondary", "type": "object"},
                {"name": "flag", "type": "boolean"},
            ],
            "return_type": "int",
            "body": body,
            "covers": ["resources_and_ui.interactions"],
        }],
        "initialize": [],
        "platform_modules": [],
        "event_bindings": [],
    }
    java = render_typed_plan_java(plan)
    assert "java.util.Objects.equals(event, " in java
    assert "if ((event ==" not in java


def test_existing_saved_plan_repairs_only_missing_ui_action_handlers(monkeypatch):
    from types import SimpleNamespace
    from minecraft_mod_ai import authored_structured_design
    from minecraft_mod_ai import authored_ui_action_repair, typed_plan_authoring
    from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir

    actions = ("buy_part", "upgrade_part")

    def concerns(_sections, section):
        if section == "algorithm":
            return {"atomic_mutations": [{"mutations": "spend_credits"}]}
        if section == "resources_and_ui":
            return {"interactions": [
                {"server_action": action, "trigger": action} for action in actions
            ]}
        if section == "integration":
            return {"entry_points": [
                {"trigger": "command:shipyard", "owner": "market", "boundary": "server"}
            ]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    monkeypatch.setattr(authored_ui_action_repair, "active_concern_records", concerns)
    monkeypatch.setattr(typed_plan_authoring, "author_semantic_game_dispatch",
                        lambda *_a, **_k: [
                            *[{
                                "op": "if",
                                "condition": {
                                    "op": "binary", "operator": "==",
                                    "left": {"op": "ref", "name": "event"},
                                    "right": {"op": "literal", "type": "string", "value": action},
                                },
                                "then": [{
                                    "op": "state_set",
                                    "key": {"op": "literal", "type": "string", "value": "credits"},
                                    "value": {"op": "literal", "type": "int", "value": index},
                                    "context": {"op": "map", "entries": []},
                                }],
                                "else": [],
                            } for index, action in enumerate(actions, start=1)],
                            {"op": "return", "value": {"op": "literal", "type": "int", "value": 1}},
                        ])
    original = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "0" * 64,
        "functions": [{
            "id": "existing_command",
            "parameters": [{"name": "source", "type": "object"}],
            "return_type": "int",
            "covers": ["integration.entry_points"],
            "body": [{"op": "return", "value": {"op": "literal", "type": "int", "value": 1}}],
        }],
        "initialize": [], "platform_modules": [],
        "event_bindings": [{
            "event": "command", "function": "existing_command",
            "entry_point_index": 0,
            "config": {"literal": "shipyard", "permission_level": 0},
        }],
    }
    original = validate_typed_plan_ir(original)
    before = deepcopy(original)
    planned = SimpleNamespace(structured_sections={}, typed_plan_ir=original)
    assert authored_ui_action_repair.authored_ui_action_repair_needed(planned)
    repaired = authored_ui_action_repair.repair_authored_ui_action_ir(
        None, "purchase and upgrade ships", {}, original, capabilities={},
    )
    assert original == before
    assert repaired["functions"][0] == before["functions"][0]
    assert repaired["event_bindings"][0] == before["event_bindings"][0]
    assert len(repaired["event_bindings"]) == 3
    assert repaired["functions"][1]["id"] == "ui_action_dispatch"
    assert repaired["functions"][2]["body"][0]["value"]["function"] == "ui_action_dispatch"
    assert not authored_ui_action_repair.authored_ui_action_repair_needed(
        SimpleNamespace(structured_sections={}, typed_plan_ir=repaired)
    )


def test_semantic_dispatch_preserves_zero_increment_without_reauthoring(monkeypatch):
    from minecraft_mod_ai import typed_plan_authoring, authored_structured_design

    def concerns(_sections, section):
        if section == "state_model":
            return {"variables": [
                {"name": "energy", "type": "int"},
                {"name": "ship_hull", "type": "int"},
            ]}
        if section == "resources_and_ui":
            return {"interactions": [{"server_action": "change_status"}]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    answers = iter([
        {"trigger_event": "command", "action_kind": "increment_state",
         "state_key": "energy", "value": 0},
        {"trigger_event": "command", "action_kind": "increment_state",
         "state_key": "ship_hull", "value": 2},
    ])
    schemas = []

    def fake_generate(*_args, **kwargs):
        schemas.append(kwargs["response_schema"])
        return next(answers)

    monkeypatch.setattr(
        typed_plan_authoring, "generate_fixed_template_value", fake_generate,
    )
    result = typed_plan_authoring.author_semantic_game_dispatch(
        None, "increase energy", {}, {},
        coverage_refs=("resources_and_ui.interactions",),
        bound_events=("command",),
    )
    assert len(schemas) == 1
    assert result[0]["then"][0]["op"] == "state_set"
    assert result[0]["then"][0]["value"]["operator"] == "+"
    assert result[0]["then"][0]["key"]["value"] == "energy"
    assert result[0]["then"][0]["value"]["right"]["value"] == 0


def test_zero_increment_remains_a_valid_typed_state_operation():
    from minecraft_mod_ai.typed_plan_authoring import (
        lower_semantic_game_dispatch_to_ir,
    )

    body = lower_semantic_game_dispatch_to_ir([{
        "trigger_event": "command",
        "action_kind": "increment_state",
        "state_key": "energy",
        "value": 0,
    }], {"energy": "int"}, {})
    write = body[0]["then"][0]
    assert write["op"] == "state_set"
    assert write["value"]["left"]["op"] == "state_get"
    assert write["value"]["right"]["value"] == 0

def test_generic_semantic_dispatch_does_not_special_case_purchase_keywords(monkeypatch):
    from minecraft_mod_ai import typed_plan_authoring, authored_structured_design

    def concerns(_sections, section):
        if section == "state_model":
            return {"variables": [{"name": "ship_hull", "type": "int"}]}
        if section == "resources_and_ui":
            return {"interactions": [{"server_action": "ship_part_purchase"}]}
        return {}

    monkeypatch.setattr(authored_structured_design, "active_concern_records", concerns)
    calls = []
    def model(_router, _role, _messages, *, response_schema, **_kwargs):
        calls.append(response_schema)
        return {
            "trigger_event": "command", "action_kind": "increment_state",
            "state_key": "ship_hull", "value": 1,
        }

    monkeypatch.setattr(typed_plan_authoring, "generate_fixed_template_value", model)
    result = typed_plan_authoring.author_semantic_game_dispatch(
        None, "buy a ship part", {}, {},
        coverage_refs=("resources_and_ui.interactions",),
        bound_events=("command",),
    )
    assert len(calls) == 1
    assert result[0]["condition"]["right"]["value"] == "ship_part_purchase"
    assert result[0]["then"][0]["op"] == "state_set"


def test_semantic_schema_has_only_declared_states_and_capabilities():
    from minecraft_mod_ai.typed_plan_authoring import semantic_dispatch_schema

    schema = semantic_dispatch_schema(
        {"health": "int"}, {}, allowed_events=("command",), mutation_only=True,
    )
    assert {branch["properties"]["action_kind"]["const"]
            for branch in schema["oneOf"]} == {"set_state", "increment_state"}


def test_repeated_typed_statements_are_rejected_before_twelve_attempts():
    from minecraft_mod_ai.typed_plan_authoring import TypedOperationAuthor
    import json

    model = NativePlanner([
        {"op": "expr"}, {"op": "literal", "type": "int", "value": 1},
        {"op": "expr"}, {"op": "literal", "type": "int", "value": 1},
    ])
    author = TypedOperationAuthor(model, "resource action", {}, {}, max_calls=10)
    with pytest.raises(ValueError, match="TYPED_PLAN_REPEATED_NO_PROGRESS"):
        author.body("ui_action.example", max_statements=12)
    assert len(model.calls) == 4
    third_request = json.loads(model.calls[2][1][1]["content"])
    assert third_request["body_progress"]["completed_count"] == 1


@pytest.fixture
def raw_semantic_concerns(monkeypatch):
    """Isolate semantic preflight from the unrelated full worksheet envelope.

    Production validates the complete canonical worksheet before this pass;
    these unit fixtures intentionally contain only the relevant concern rows.
    """
    from minecraft_mod_ai import authored_structured_design

    monkeypatch.setattr(
        authored_structured_design, "active_concern_records",
        lambda sections, name: sections.get(name, {}).get("specification", {}),
    )


def test_ui_only_metadata_without_mutation_fails_closed(raw_semantic_concerns):
    from minecraft_mod_ai.typed_action_authoring import preflight_ui_action_effects
    ui = {"resources_and_ui": {"specification": {"interactions": [{
        "trigger": "Open builder screen",
        "server_action": {"intent": "Ship construction UI", "inputs": []},
        "client_feedback": "Show available sections",
    }]}}}
    ui["state_model"] = {"specification": {
        "variables": [{"name": "credits", "type": "string"}],
        "updates": [{"mutation": [{
            "target": "credits", "operator": "=",
            "value": {"kind": "state_ref", "name": "credits"},
        }]}],
    }}
    with pytest.raises(ValueError, match="ACTION_DESIGN_NO_EXECUTABLE_EFFECT"):
        preflight_ui_action_effects(ui)
    ui["state_model"]["specification"]["updates"][0]["mutation"][0] = {
        "target": "credits", "operator": "-=",
        "value": {"kind": "literal", "type": "int", "value": 10},
    }
    # A real write belonging to another trigger still cannot implement this UI.
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(ui)
    ui["state_model"]["specification"]["updates"][0]["trigger"] = "Open builder screen"
    preflight_ui_action_effects(ui)


def test_state_guard_preflight_rejects_self_comparison_and_text_ordering(raw_semantic_concerns):
    from minecraft_mod_ai.typed_action_authoring import preflight_state_guard_semantics
    left = {"kind": "state_ref", "name": "player_credits"}
    sections = {"state_model": {"specification": {
        "variables": [{"name": "player_credits", "type": "string"}],
        "transitions": [{"guard": {"kind": "compare", "op": ">=",
                                   "left": left, "right": left}, "mutation": []}],
    }}}
    with pytest.raises(ValueError, match="STATE_GUARD_TAUTOLOGY"):
        preflight_state_guard_semantics(sections)
    sections["state_model"]["specification"]["transitions"][0]["guard"]["right"] = {
        "kind": "literal", "type": "int", "value": 10,
    }
    with pytest.raises(ValueError, match="STATE_GUARD_NONNUMERIC_ORDERING"):
        preflight_state_guard_semantics(sections)
    sections["state_model"]["specification"]["variables"][0]["type"] = "int"
    preflight_state_guard_semantics(sections)


def test_state_preflight_rejects_identity_mutations_across_state_collections(raw_semantic_concerns):
    from minecraft_mod_ai.typed_action_authoring import preflight_state_guard_semantics

    unchanged = {
        "target": "player_credits", "operator": "=",
        "value": {"kind": "state_ref", "name": "player_credits"},
    }
    cases = (
        ("transitions", "mutation"),
        ("updates", "mutation"),
        ("initialization", "initial_state"),
        ("cleanup", "action"),
    )
    for collection, field in cases:
        sections = {"state_model": {"specification": {
            "variables": [{"name": "player_credits", "type": "int"}],
            collection: [{field: [unchanged]}],
        }}}
        with pytest.raises(ValueError, match="STATE_IDENTITY_MUTATION"):
            preflight_state_guard_semantics(sections)


def test_state_identity_preflight_on_canonical_valid_worksheet():
    """The real normalization path admits the shape but rejects the no-op."""
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
    from minecraft_mod_ai.typed_action_authoring import preflight_state_guard_semantics

    specification = {concern: [] for concern in DETAIL_RECORDS["state_model"]}
    specification["inapplicable_concerns"] = []
    specification["variables"] = [{
        "name": "player_credits", "owner": "player", "type": "int",
        "unit": "credits", "default": "0", "domain": "0..1000000",
    }]
    specification["updates"] = [{
        "trigger": "buy", "owner": "server",
        "mutation": [{
            "target": "player_credits", "operator": "=",
            "value": {"kind": "state_ref", "name": "player_credits"},
        }],
    }]
    sections = {"state_model": {
        "specification": specification, "constraint_evidence_refs": [],
    }}
    with pytest.raises(ValueError, match="STATE_IDENTITY_MUTATION"):
        preflight_state_guard_semantics(sections)
