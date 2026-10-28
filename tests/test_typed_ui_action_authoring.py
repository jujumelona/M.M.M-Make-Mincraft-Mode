"""A concrete UI action owns inputs and one typed server body."""

from copy import deepcopy
import json
import shutil
import subprocess

import pytest

from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts


def sections(server_action=None):
    from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS

    result = {
        name: {
            "constraint_evidence_refs": [],
            "specification": {
                **{key: [] for key in DETAIL_RECORDS[name]},
                "inapplicable_concerns": [],
            },
        }
        for name in ("resources_and_ui", "state_model", "integration", "algorithm")
    }
    result["resources_and_ui"]["specification"]["interactions"] = [
        {
            "trigger": "User submits the existing selection form",
            "server_action": server_action
            or {
                "intent": "Record the requested amount",
                "inputs": [
                    {
                        "name": "amount",
                        "type": "int",
                        "label": "Amount",
                        "minimum": 1,
                        "maximum": 10,
                    }
                ],
            },
            "client_feedback": "Show the current amount",
        }
    ]
    result["state_model"]["specification"]["variables"] = [
        {
            "name": "amount",
            "type": "int",
            "default": "0",
            "owner": "player",
            "domain": "count",
            "unit": "count",
        }
    ]
    result["state_model"]["specification"]["updates"] = [
        {
            "owner": "player",
            "trigger": "User submits the existing selection form",
            "mutation": [
                {
                    "target": "amount",
                    "operator": "=",
                    "value": {"kind": "context_ref", "name": "amount"},
                }
            ],
        }
    ]
    return result


def state_body():
    return [
        {
            "op": "state_set",
            "key": {"op": "literal", "type": "string", "value": "amount"},
            "value": {"op": "ref", "name": "amount"},
            "context": {"op": "map", "entries": []},
        },
        {"op": "return", "value": {"op": "literal", "type": "int", "value": 1}},
    ]


def test_one_action_body_receives_typed_inputs_and_its_own_intent(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    seen = []

    def body(author, scope, *, env, **kwargs):
        current = author._scope_semantic_context(scope)
        record = current["resources_and_ui"]["interactions"][0]
        seen.append(record)
        assert env["amount"] == "int"
        assert record["server_action"]["inputs"][0]["name"] == "amount"
        return state_body()

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body", body
    )
    functions, bindings = author_action_functions(
        None,
        "Use a selection form",
        sections(),
        typed_host_capability_contracts(),
        first_index=2,
    )
    plan = validate_typed_plan_ir(
        {
            "schema_version": "mmm/typed-plan-ir-v1",
            "source_sha256": "b" * 64,
            "functions": functions,
            "event_bindings": bindings,
            "initialize": [],
        },
        capabilities=typed_host_capability_contracts(),
    )
    assert len(seen) == 1
    assert functions[0]["parameters"][1] == {"name": "amount", "type": "int"}
    assert bindings[0]["entry_point_index"] == 2
    assert plan["event_bindings"][0]["function"] == functions[0]["id"]
    assert "ui_action_dispatch" not in json.dumps(plan)


@pytest.mark.parametrize("bad", ["message", "ignored_input", "wrong_state_type"])
def test_bounded_author_cannot_complete_ui_action_with_unrelated_or_invalid_body(
    monkeypatch, bad
):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    def body(*args, **kwargs):
        if bad == "message":
            return [
                {"op": "return", "value": {"op": "literal", "type": "int", "value": 1}}
            ]
        result = state_body()
        if bad == "ignored_input":
            result[0]["value"] = {"op": "literal", "type": "int", "value": 1}
        if bad == "wrong_state_type":
            result[0]["value"] = {"op": "literal", "type": "string", "value": "bad"}
        return result

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body", body
    )
    with pytest.raises(ValueError, match="ACTION_"):
        author_action_functions(
            None,
            "Use a selection form",
            sections(),
            typed_host_capability_contracts(),
            first_index=0,
        )


def test_saved_prose_ui_migrates_single_action_and_preserves_non_ui_functions(
    monkeypatch,
):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.typed_action_authoring import migrate_ui_action_plan

    old = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "b" * 64,
        "initialize": [],
        "event_bindings": [],
        "functions": [
            {
                "id": "unrelated",
                "parameters": [],
                "return_type": "void",
                "body": [],
                "covers": [],
            }
        ],
    }
    design = sections("Record the selected amount on the server")
    expected = sections()["resources_and_ui"]["specification"]["interactions"][0][
        "server_action"
    ]
    calls = []

    def metadata(router, record, **kwargs):
        calls.append(record["server_action"])
        return expected

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_action_authoring.author_action_metadata", metadata
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: state_body(),
    )
    saved = AuthoredPlan(
        "Use a selection form",
        "Use a selection form",
        structured_sections=design,
        typed_plan_ir=old,
    )
    before = deepcopy(saved.to_dict())
    result = migrate_ui_action_plan(None, saved)
    assert saved.to_dict() == before
    assert result.typed_plan_ir["functions"][0] == old["functions"][0]
    assert len(result.typed_plan_ir["event_bindings"]) == 1
    assert (
        result.structured_sections["resources_and_ui"]["specification"]["interactions"][
            0
        ]["server_action"]
        == expected
    )
    assert calls == ["Record the selected amount on the server"]
    assert migrate_ui_action_plan(None, result).to_dict() == result.to_dict()


def test_migration_preserves_an_unrelated_alias_of_a_legacy_ui_function(monkeypatch):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.typed_action_authoring import migrate_ui_action_plan

    ir = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "b" * 64,
        "initialize": [],
        "functions": [
            {
                "id": "uiAction1_command",
                "parameters": [{"name": "source", "type": "object"}],
                "return_type": "int",
                "covers": [],
                "body": [
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "int", "value": 1},
                    }
                ],
            }
        ],
        "event_bindings": [
            {
                "event": "command",
                "function": "uiAction1_command",
                "entry_point_index": i,
                "config": {"literal": literal, "permission_level": 0},
            }
            for i, literal in enumerate(("mmm_action_1", "admin_existing"))
        ],
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_action_authoring.author_action_metadata",
        lambda *a, **k: sections()["resources_and_ui"]["specification"]["interactions"][
            0
        ]["server_action"],
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: state_body(),
    )
    saved = AuthoredPlan(
        "Use form",
        "Use form",
        structured_sections=sections("Set selected amount"),
        typed_plan_ir=ir,
    )
    migrated = migrate_ui_action_plan(None, saved)
    assert "admin_existing" in [
        b["config"]["literal"] for b in migrated.typed_plan_ir["event_bindings"]
    ]
    assert ir["functions"][0] in migrated.typed_plan_ir["functions"]


def test_foreach_input_variable_is_admitted_in_its_nested_scope(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    body = state_body()
    body[0]["value"] = {"op": "ref", "name": "item"}
    loop = {
        "op": "foreach",
        "name": "item",
        "type": "int",
        "collection": {"op": "list", "items": [{"op": "ref", "name": "amount"}]},
        "body": [body[0]],
    }
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: [loop, body[1]],
    )
    functions, _ = author_action_functions(
        None, "Use form", sections(), typed_host_capability_contracts(), first_index=0
    )
    assert functions[0]["body"][0]["op"] == "foreach"


def test_action_context_cannot_be_an_imported_client_input():
    from minecraft_mod_ai.typed_action_authoring import validate_action_metadata

    metadata = sections()["resources_and_ui"]["specification"]["interactions"][0][
        "server_action"
    ]
    metadata["inputs"][0]["name"] = "action_context"
    with pytest.raises(ValueError, match="ACTION_INPUT_RESERVED"):
        validate_action_metadata(metadata)


def test_older_targets_preserve_legacy_ui_without_new_model_calls(monkeypatch):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.typed_action_authoring import (
        migrate_ui_action_plan,
        promote_action_sections,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported target must be resolved before model calls")

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_action_authoring.author_action_metadata", unexpected
    )
    design = sections("Set selected amount")
    plan = AuthoredPlan(
        "Use form", "Use form", structured_sections=design, typed_plan_ir={}
    )
    for version in ("1.21.5", "26.1.2"):
        assert (
            promote_action_sections(None, design, minecraft_version=version) is design
        )
        assert migrate_ui_action_plan(None, plan, minecraft_version=version) is plan
        with pytest.raises(ValueError, match="ACTION_CLIENT_TARGET_UNSUPPORTED"):
            promote_action_sections(None, sections(), minecraft_version=version)


def test_normal_typed_authoring_uses_action_function_as_the_ui_runtime_root(
    monkeypatch,
):
    from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues

    design = sections()
    owners = [
        {
            "feature_id": "__typed_ui__",
            "owner_module": "authored_typed_plan",
            "ui_action": action_record_identity(
                design["resources_and_ui"]["specification"]["interactions"][0]
            ),
            "worksheet_record_refs": ["state_model.updates[0]"],
        }
    ]
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: state_body(),
    )
    plan = author_typed_plan_ir(
        None,
        "Use a selection form",
        design,
        typed_host_capability_contracts(),
        runtime_owners=owners,
    )
    assert len(plan["event_bindings"]) == 1
    assert "action" in plan["event_bindings"][0]["config"]
    assert typed_plan_support_issues(design, plan, runtime_owners=owners) == ()


def test_legacy_input_abi_is_authored_as_separate_bounded_decisions():
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata

    responses = [
        {"label": "Set quantity"},
        {"count": 1},
        {"type": "int", "label": "Quantity"},
        {"minimum": 1, "maximum": 10},
    ]
    seen = []

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            payload = json.loads(messages[-1]["content"])
            assert "interaction" in payload
            assert len(parameters["properties"]) <= 2
            seen.append(data_name)
            return responses.pop(0)

    metadata = author_action_metadata(
        Router(),
        {"trigger": "submit form", "server_action": "Set the selected quantity"},
    )
    assert metadata["inputs"] == [
        {
            "name": "input_1",
            "type": "int",
            "label": "Quantity",
            "minimum": 1,
            "maximum": 10,
        }
    ]
    assert seen == [
        "ui_action_label",
        "ui_action_input_count",
        "ui_action_input_type",
        "ui_action_input_bounds",
    ]


def test_double_input_bounds_use_bounded_decimal_text_before_host_conversion():
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata

    responses = [
        {"label": "Set value"},
        {"count": 1},
        {"type": "double", "label": "Value"},
        {"minimum": "-1.25e2", "maximum": "2.5e3"},
    ]

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if data_name == "ui_action_input_bounds":
                assert parameters["properties"]["minimum"]["maxLength"] == 24
            return responses.pop(0)

    metadata = author_action_metadata(
        Router(), {"trigger": "submit", "server_action": "Set supplied value"}
    )
    assert metadata["inputs"][0]["minimum"] == -125.0
    assert metadata["inputs"][0]["maximum"] == 2500.0


@pytest.mark.parametrize("read", [("missing", "int"), ("amount", "string")])
def test_action_reads_must_match_declared_state_contract(monkeypatch, read):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    body = state_body()
    body.insert(
        0,
        {
            "op": "let",
            "name": "snapshot",
            "type": read[1],
            "value": {
                "op": "state_get",
                "key": {"op": "literal", "type": "string", "value": read[0]},
                "type": read[1],
                "context": {"op": "map", "entries": []},
            },
        },
    )
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: body,
    )
    with pytest.raises(ValueError, match="ACTION_STATE_"):
        author_action_functions(
            None,
            "Use form",
            sections(),
            typed_host_capability_contracts(),
            first_index=0,
        )


def test_identical_repeated_ui_records_share_one_actual_server_binding(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    design = sections()
    rows = design["resources_and_ui"]["specification"]["interactions"]
    rows.append(deepcopy(rows[0]))
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: state_body(),
    )
    functions, bindings = author_action_functions(
        None,
        "Use a selection form",
        design,
        typed_host_capability_contracts(),
        first_index=0,
    )
    assert len(functions) == len(bindings) == 1


def test_host_supplies_authenticated_state_context_to_the_bounded_body(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    def body(author, scope, *, env, **kwargs):
        assert env["action_context"] == "map"
        return state_body()

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body", body
    )
    functions, _ = author_action_functions(
        None, "Use form", sections(), typed_host_capability_contracts(), first_index=0
    )
    context = functions[0]["body"][0]["context"]
    assert context["entries"][0]["key"]["value"] == "player"
    assert context["entries"][0]["value"]["id"] == "player.uuid"
    assert context["entries"][0]["value"]["args"] == [{"op": "ref", "name": "source"}]


def test_encoded_action_mutates_the_real_player_state_owner(monkeypatch, tmp_path):
    from minecraft_mod_ai.production_state_compiler import render_production_state_java
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.typed_action_java import render_action_dispatch_java
    from minecraft_mod_ai.typed_plan_java import render_typed_plan_java

    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: state_body(),
    )
    design = sections()
    functions, bindings = author_action_functions(
        None, "Use form", design, typed_host_capability_contracts(), first_index=0
    )
    plan = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "a" * 64,
        "functions": functions,
        "event_bindings": bindings,
        "initialize": [],
    }
    identity = bindings[0]["config"]["action"]["id"]
    source_plan = {**plan, "event_bindings": []}
    sources = {
        "AuthoredProgram.java": render_typed_plan_java(
            source_plan, package="proof", capabilities=typed_host_capability_contracts()
        ),
        "AuthoredActions.java": render_action_dispatch_java(plan, package="proof"),
        "AuthoredStateModel.java": render_production_state_java(
            design["state_model"], package_name="proof"
        ),
        "AuthoredHostCapabilities.java": "package proof; public class AuthoredHostCapabilities { public static String playerUuid(Object actor) { return actor.toString(); } }",
        "Probe.java": '''package proof;
public class Probe {
 static void check(boolean value) { if (!value) throw new AssertionError(); }
 public static void main(String[] args) {
  String id = "'''
        + identity
        + """";
  var a = java.util.Map.<String,Object>of("player", "player-a");
  var b = java.util.Map.<String,Object>of("player", "player-b");
  String command = AuthoredActions.commandFromText(id, java.util.Map.of("amount", "3"));
  check(AuthoredActions.invokeEncoded(id, "player-a", java.util.List.of(command.split(" ")[1])) == 1);
  AuthoredActions.invoke(id, "player-b", java.util.Map.of("amount", 7));
  check(AuthoredStateModel.readState_amount(a) == 3 && AuthoredStateModel.readState_amount(b) == 7);
  check(AuthoredActions.submitEncoded(id, "player-a", java.util.List.of("xMTE")) == 0);
  check(AuthoredStateModel.readState_amount(a) == 3 && AuthoredStateModel.readState_amount(b) == 7);
 }
}""",
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(source)
    compiled = subprocess.run(
        [javac, "-d", str(tmp_path), *map(str, tmp_path.glob("*.java"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "proof.Probe"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert executed.returncode == 0, executed.stderr


def test_production_repeated_ui_title_and_options_do_not_fabricate_enum_or_inputs():
    """Replay the real input_count=16 / repeated-title failure as bounded decisions."""
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata

    title = "Ground-mode resource and ship-construction UI"
    answers = [
        {"label": title},
        {"count": 4},  # 16 is no longer a legal prose-derived control count.
        {"type": "string", "label": title},
        {"type": "string", "label": "Selected purchase or upgrade"},
        {"type": "string", "label": title},
        {"type": "string", "label": "Selected purchase or upgrade"},
    ]
    requested = []

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            requested.append((data_name, json.loads(messages[-1]["content"]), parameters))
            return answers.pop(0)

    result = author_action_metadata(Router(), {
        "trigger": "Player opens the ground-mode resource and ship-construction UI",
        "server_action": "Validate the selected purchase or upgrade and apply its state",
        "client_feedback": "Show the selected section, credits and result",
    })
    assert answers == []
    assert len(result["inputs"]) == 1
    assert result["inputs"][0] == {
        "name": "input_1", "type": "string",
        "label": "Selected purchase or upgrade", "max_length": 160,
        "choices": [],
    }
    assert requested[1][2]["properties"]["count"]["maximum"] == 4
    assert not any(name == "ui_action_input_choices" for name, _, _ in requested)
    assert requested[3][1]["rejected_input"]["label"] == title


def test_repeated_screen_title_is_an_explicit_missing_input_not_a_fake_field():
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata
    title = "Shipyard screen"
    answers = [
        {"label": title}, {"count": 2},
        {"type": "string", "label": title},
        {"type": "string", "label": title},
    ]

    class Router:
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            return answers.pop(0)

    with pytest.raises(ValueError, match="ACTION_INPUT_NOT_A_PLAYER_VALUE"):
        author_action_metadata(Router(), {
            "trigger": "Player selects an upgrade",
            "server_action": "Upgrade selected equipment",
            "client_feedback": "Show equipment",
        })


def test_structured_ui_enum_remains_explicit_and_is_never_replaced_by_model_values():
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata

    metadata = {
        "intent": "Choose ship section",
        "inputs": [{
            "name": "section", "type": "string", "label": "Section",
            "max_length": 16, "choices": ["hull", "weapon", "crew"],
        }],
    }

    class NoInference:
        def generate_structured_data(self, *a, **kw):
            raise AssertionError("Declared enum is authoritative; never reinfer it")

    got = author_action_metadata(NoInference(), {
        "trigger": "Select ship section", "server_action": metadata,
    })
    assert got == metadata


def test_repeated_nested_ast_recovers_only_exact_source_owned_ui_mutation(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    calls = []

    def stuck_body(*args, **kwargs):
        calls.append(kwargs["return_type"])
        raise ValueError(
            "TYPED_PLAN_REPEATED_NO_PROGRESS: "
            "ui_action.action_sample.statement[0].then repeated after 2 statements"
        )

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        stuck_body,
    )
    functions, bindings = author_action_functions(
        None, "Use a selection form", sections(),
        typed_host_capability_contracts(), first_index=0,
    )
    assert calls == ["int"]
    assert len(functions) == len(bindings) == 1
    body = functions[0]["body"]
    assert body[0]["op"] == "state_set"
    assert body[0]["key"]["value"] == "amount"
    assert body[0]["value"] == {"op": "ref", "name": "amount"}
    assert body[0]["context"]["op"] == "map"
    assert body[-1] == {
        "op": "return",
        "value": {"op": "literal", "type": "int", "value": 1},
    }
    validate_typed_plan_ir({
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": "0" * 64,
        "functions": functions,
        "event_bindings": bindings,
        "initialize": [],
    }, capabilities=typed_host_capability_contracts())


def test_repeated_ast_never_guesses_missing_or_ambiguous_ui_effect(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    def stuck_body(*args, **kwargs):
        raise ValueError(
            "TYPED_PLAN_REPEATED_NO_PROGRESS: ui_action.sample.then"
        )

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        stuck_body,
    )
    for change, error in (
        ("missing", "ACTION_HOST_LOWERING_REQUIRED"),
        ("ambiguous", "ACTION_HOST_LOWERING_REQUIRED"),
        ("unknown_input", "ACTION_UI_VALUE_ABI_UNBOUND: ACTION_HOST_INPUT_UNBOUND: not_a_client_input"),
    ):
        design = sections()
        updates = design["state_model"]["specification"]["updates"]
        if change == "missing":
            updates.clear()
        elif change == "ambiguous":
            updates.append(deepcopy(updates[0]))
        else:
            updates[0]["mutation"][0]["value"]["name"] = "not_a_client_input"
        with pytest.raises(ValueError, match=error):
            author_action_functions(
                None, "Use a selection form", design,
                typed_host_capability_contracts(), first_index=0,
            )


def test_repeated_ast_recovery_preserves_explicit_ui_ownership(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("TYPED_PLAN_SEMANTIC_BUDGET_EXHAUSTED: ui_action")
        ),
    )
    design = sections()
    design["state_model"]["specification"]["updates"].append(
        deepcopy(design["state_model"]["specification"]["updates"][0])
    )
    action = design["resources_and_ui"]["specification"]["interactions"][0]
    owners = [{
        "feature_id": "__typed_ui__",
        "ui_action": action_record_identity(action),
        "worksheet_record_refs": ["state_model.updates[1]"],
    }]
    functions, _ = author_action_functions(
        None, "Use a selection form", design,
        typed_host_capability_contracts(), first_index=0,
        runtime_owners=owners,
    )
    assert len(functions) == 1
    assert [row["op"] for row in functions[0]["body"]] == [
        "state_set", "return",
    ]


def test_production_rejects_unrelated_launch_write_as_ui_transaction():
    from minecraft_mod_ai.typed_action_authoring import preflight_ui_action_effects

    design = sections()
    interaction = design["resources_and_ui"]["specification"]["interactions"][0]
    interaction["trigger"] = (
        "Player opens the ground-mode resource and ship-construction UI"
    )
    interaction["server_action"]["intent"] = (
        "Ground-mode resource and ship-construction UI"
    )
    # A real write exists, but it belongs to another trigger.
    updates = design["state_model"]["specification"]["updates"]
    assert updates[0]["mutation"][0]["target"] == "amount"
    assert updates[0]["trigger"] != interaction["trigger"]
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(design)


def test_production_never_generates_recursive_ast_for_missing_transaction(
    monkeypatch,
):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    design = sections()
    interaction = design["resources_and_ui"]["specification"]["interactions"][0]
    interaction["server_action"]["intent"] = "Purchase a ship upgrade"
    interaction["trigger"] = "Purchase selected ship upgrade"
    # An unrelated credit reset is not authorization for the purchase.
    design["state_model"]["specification"]["updates"][0]["trigger"] = "launch_to_space"

    def forbidden_model_call(*args, **kwargs):
        raise AssertionError(
            "production may not fall back to recursive model AST invention"
        )

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        forbidden_model_call,
    )
    with pytest.raises(ValueError, match="ACTION_SOURCE_EFFECT_UNBOUND"):
        author_action_functions(
            None, "Construct and purchase ship upgrades", design,
            typed_host_capability_contracts(), first_index=0,
            prefer_grounded=True,
        )


def test_production_direct_lowering_preserves_matched_input_effect(monkeypatch):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    monkeypatch.setattr(
        "minecraft_mod_ai.typed_plan_authoring.TypedOperationAuthor.body",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("grounded action must not spend a model call")
        ),
    )
    functions, bindings = author_action_functions(
        None, "Use the selection form", sections(),
        typed_host_capability_contracts(), first_index=0,
        prefer_grounded=True,
    )
    assert len(functions) == len(bindings) == 1
    assert functions[0]["body"][0]["op"] == "state_set"
    assert functions[0]["body"][0]["value"] == {
        "op": "ref", "name": "amount",
    }


def test_prose_action_keeps_server_semantics_instead_of_screen_title():
    from minecraft_mod_ai.typed_action_authoring import author_action_metadata

    original = "Resource_economy_service validates the purchase and changes credits"
    answers = [{"label": "Ground-mode resource and ship-construction UI"}, {"count": 0}]
    class Router:
        def generate_structured_data(self, *args, **kwargs):
            return answers.pop(0)

    metadata = author_action_metadata(Router(), {
        "trigger": "Purchase upgrade",
        "server_action": original,
        "client_feedback": "Show the shipyard screen",
    })
    assert metadata == {"intent": original, "inputs": []}
    assert not answers


def test_production_checks_all_ui_actions_for_source_owned_mutation():
    from minecraft_mod_ai.typed_action_authoring import preflight_ui_action_effects

    design = sections()
    rows = design["resources_and_ui"]["specification"]["interactions"]
    rows[0]["server_action"]["intent"] = "Purchase an upgrade from resource service"
    preflight_ui_action_effects(design, strict_source=True)

    # The logged design contains a real credits write, but at launch rather
    # than when the purchase UI is submitted.
    design["state_model"]["specification"]["updates"][0]["trigger"] = "launch_to_space"
    with pytest.raises(ValueError, match="ACTION_UI_EFFECT_UNBOUND"):
        preflight_ui_action_effects(design, strict_source=True)


def test_owner_assignment_cannot_borrow_another_trigger_effect():
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity

    design = sections()
    action = design["resources_and_ui"]["specification"]["interactions"][0]
    design["state_model"]["specification"]["updates"][0]["trigger"] = "launch_to_space"
    owners = [{
        "feature_id": "__typed_ui__",
        "ui_action": action_record_identity(action),
        "worksheet_record_refs": ["state_model.updates[0]"],
    }]
    with pytest.raises(ValueError, match="ACTION_HOST_OWNER_TRIGGER_MISMATCH"):
        author_action_functions(
            None, "Purchase an upgrade", design,
            typed_host_capability_contracts(), first_index=0,
            runtime_owners=owners, prefer_grounded=True,
        )


def test_duplicate_credited_transactions_are_not_applied_twice():
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.authored_ui_action_bindings import action_record_identity

    design = sections()
    updates = design["state_model"]["specification"]["updates"]
    updates.append(deepcopy(updates[0]))
    action = design["resources_and_ui"]["specification"]["interactions"][0]
    owners = [{
        "feature_id": "__typed_ui__",
        "ui_action": action_record_identity(action),
        "worksheet_record_refs": ["state_model.updates[0]", "state_model.updates[1]"],
    }]
    with pytest.raises(ValueError, match="ACTION_HOST_DUPLICATE_MUTATION"):
        author_action_functions(
            None, "Select the number to record", design,
            typed_host_capability_contracts(), first_index=0,
            runtime_owners=owners, prefer_grounded=True,
        )
