from copy import deepcopy

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
            "int_value": "7",
            "capability_id": "",
            "message": "",
        },
        {
            "trigger_event": "any",
            "state_key": "credits",
            "action_kind": "set_state",
            "int_value": "10",
            "capability_id": "",
            "message": "",
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
        import re

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
            # When the effective decoder schema drops pattern (llama.cpp
            # precedence fix), maxLength is the only remaining hint.
            max_len = s.get("maxLength", 0)
            if not pat and isinstance(max_len, int):
                # Transport-encoded integer (maxLength=20) or number (maxLength=32)
                if max_len == 20 or max_len == 32:
                    return "1"
                # Resource ID fields use maxLength >= 64
                if max_len >= 64:
                    return "minecraft:stone"
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
            covers = ["authority_and_network.packets"]
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



