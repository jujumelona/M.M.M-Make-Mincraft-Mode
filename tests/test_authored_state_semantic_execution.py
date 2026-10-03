from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
from minecraft_mod_ai.authored_structured_design import structured_sections_sha256
from minecraft_mod_ai.canonical_concern_authority import CanonicalConcernAuthority
from minecraft_mod_ai.implementation_graph_execution import (
    _bind_atomic_leaf_contract,
    _normalize_implementation_graph_request,
)


def state_request():
    # The failing transition is copied from the supplied production trace.
    section = {"specification": {
        "variables": [
            {"name": "player_currency", "owner": "Player", "type": "integer",
             "unit": "credits", "default": "100", "domain": "0 to infinity"},
            {"name": "ship_blueprint", "owner": "Player", "type": "item",
             "unit": "blueprint", "default": "0", "domain": "1 to 100"},
        ],
        "transitions": [{
            "from_state": "resource_farming_idle",
            "trigger": "player_interact_with_trade_station",
            "guard": "player_currency >= 0 AND ship_blueprint != null",
            "mutation": "update_player_currency(player_currency - trade_cost); update_ship_blueprint(ship_blueprint)",
            "to_state": "currency_acquired",
        }],
        "invariants": [{"condition": "The player's currency must never be negative",
                        "enforcement": "reject negative balances"}],
        "initialization": [], "updates": [], "cleanup": [], "concurrency": [],
        "inapplicable_concerns": [],
    }, "constraint_evidence_refs": []}
    structured = {"state_model": section}
    return {
        "text": "## state_model\n- variables: player_currency, ship_blueprint\n",
        "structured_sections": structured,
        "structured_sections_sha256": structured_sections_sha256(structured),
        "canonical_concern_authority": CanonicalConcernAuthority.from_structured_sections(structured).to_dict(),
    }


def bound_state(request):
    normalized = _normalize_implementation_graph_request(request)
    task = {"task_id": "ir_authoredstatemodel"}
    section, active = _bind_atomic_leaf_contract(
        task, {"symbol": "AuthoredStateModel", "obligations": []}, {},
        structured_sections=normalized["structured_sections"],
        canonical_concern_authority=normalized["canonical_concern_authority"],
        production_state_section=normalized["production_state_section"],
    )
    return task, section, active


def test_graph_boundary_preserves_uncompiled_actions_and_conditions():
    request = state_request()
    original = deepcopy(request)
    normalized = _normalize_implementation_graph_request(request)
    state = normalized["production_state_section"]["specification"]
    assert state["transitions"][0]["mutation"] == (
        "update_player_currency(player_currency - trade_cost); update_ship_blueprint(ship_blueprint)"
    )
    assert state["invariants"][0]["condition"] == "The player's currency must never be negative"
    assert request == original


@pytest.mark.parametrize("native", [False, True])
def test_authored_state_transition_compiles_and_changes_balance(tmp_path, native):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    request = state_request()
    original = deepcopy(request)
    task, section, active = bound_state(request)
    calls = []

    def render(work, statements, messages):
        if not native:
            return work["declaration"] + " {\n" + "\n".join(statements) + "\n}"
        from jsonschema import Draft202012Validator

        from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
        from minecraft_mod_ai.model_output_atomicity_contract import (
            assert_atomic_model_schema,
        )

        assert statements
        decisions = [{"statement": statements[0]}]
        for statement in statements[1:]:
            decisions.extend([{"part": "body"}, {"statement": statement}])
        decisions.append({"part": "done"})

        class Router:
            def generate_tool_decision(self, role, request, **kwargs):
                assert role == "coder"
                schema = kwargs["parameters"]
                assert_atomic_model_schema(schema, surface="state helper regression")
                payload = json.loads(request[-1]["content"])
                assert payload["state_lowering"]["work"][0]["record"] == work["record"]
                decision = decisions.pop(0)
                Draft202012Validator(schema).validate(decision)
                return decision

        result = _call_atomic_java_region(Router(), messages, output_token_ceiling=2048)
        assert not decisions
        return result

    def coder(messages):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload)
        work = payload["state_lowering"]["work"]
        if payload["concern"]["name"] == "transitions":
            assert len(work) == 1
            assert work[0]["field"] == "mutation"
            assert work[0]["record"]["mutation"].startswith("update_player_currency(")
            return render(work[0], [
                'double balance = ((Number) getState("player_currency", context)).doubleValue();',
                'double cost = ((Number) context.get("trade_cost")).doubleValue();',
                'setState("player_currency", balance - cost, context);',
                'setState("ship_blueprint", getState("ship_blueprint", context), context);',
            ], messages)
        assert payload["concern"]["name"] == "invariants"
        assert work[0]["field"] == "condition"
        return render(work[0], [
            'return ((Number) getState("player_currency", context)).doubleValue() >= 0;',
        ], messages)

    def compile_java(root):
        result = subprocess.run([javac, "AuthoredStateModel.java", "Probe.java"],
                                cwd=root, capture_output=True, text=True, timeout=60, check=False)
        return SimpleNamespace(status="PASS" if result.returncode == 0 else "FAIL",
                               error=result.stdout + result.stderr)

    (tmp_path / "Probe.java").write_text('''
public class Probe {
    public static void main(String[] args) {
        java.util.Map<String, Object> context = new java.util.HashMap<>();
        context.put("trade_cost", 25);
        String next = AuthoredStateModel.transition("resource_farming_idle",
                "player_interact_with_trade_station", context);
        if (!next.equals("currency_acquired")) throw new AssertionError(next);
        if (((Number) AuthoredStateModel.getState("player_currency")).doubleValue() != 75)
            throw new AssertionError("currency mutation was dropped");
        if (!AuthoredStateModel.invariantsHold(context)) throw new AssertionError("guard degraded");
        AuthoredStateModel.setState("player_currency", -1, context);
        if (AuthoredStateModel.invariantsHold(context)) throw new AssertionError("invariant dropped");
        System.out.println("balance=75; negative balance rejected");
    }
}
''', encoding="utf-8")
    executor = AtomicConcernExecutor(
        root=tmp_path, target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="public final class AuthoredStateModel { // MMM_AUTHORED_FEATURE_BODY\n}",
        task=task, section=section, concerns=active, grounding={}, dependency_source="",
        require_initialize=False, call_coder=coder, compile_java=compile_java,
        compile_log=lambda result: result.error,
        write_source=lambda path, source: path.write_text(source, encoding="utf-8"),
    )
    result = executor.run()
    assert result["repair_count"] == 0
    assert [p["concern"]["name"] for p in calls] == ["transitions", "invariants"]
    run = subprocess.run([java, "-cp", str(tmp_path), "Probe"], capture_output=True,
                         text=True, timeout=30, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "balance=75; negative balance rejected" in run.stdout
    assert request == original


def test_supported_state_still_needs_no_model(tmp_path):
    request = state_request()
    request["structured_sections"]["state_model"]["specification"]["transitions"][0]["mutation"] = "player_currency -= trade_cost"
    request["structured_sections"]["state_model"]["specification"]["invariants"][0]["condition"] = "player_currency >= 0"
    request["structured_sections_sha256"] = structured_sections_sha256(request["structured_sections"])
    request["canonical_concern_authority"] = CanonicalConcernAuthority.from_structured_sections(request["structured_sections"]).to_dict()
    task, section, active = bound_state(request)
    def forbidden(_messages):
        pytest.fail("supported DSL must stay host compiled")
    executor = AtomicConcernExecutor(
        root=tmp_path, target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="// MMM_AUTHORED_FEATURE_BODY", task=task, section=section,
        concerns=active, grounding={}, dependency_source="", require_initialize=False,
        call_coder=forbidden, compile_java=lambda _: SimpleNamespace(status="PASS"),
        compile_log=lambda _: "", write_source=lambda path, source: None,
    )
    assert executor.run()["repair_count"] == 0


@pytest.mark.parametrize("concern,field,record", [
    ("transitions", "guard", {"from_state": "idle", "trigger": "buy", "guard": "has_resources(player)", "mutation": "", "to_state": "ready"}),
    ("transitions", "mutation", {"from_state": "idle", "trigger": "buy", "guard": "true", "mutation": "player_currency -= cost; install_module(ship_blueprint)", "to_state": "ready"}),
    ("invariants", "condition", {"condition": "All pending purchases must fit the balance", "enforcement": "reject"}),
    ("initialization", "initial_state", {"owner": "Player", "trigger": "join", "initial_state": "player_currency=0, ship_blueprint={engine: null}"}),
    ("updates", "mutation", {"owner": "Player", "trigger": "trade", "mutation": "player_currency increases by transaction amount"}),
    ("cleanup", "action", {"event": "logout", "action": "release_session_resources(player)", "retained_state": "player_currency"}),
])
def test_every_uncompiled_field_keeps_whole_record_and_host_registration(concern, field, record):
    from minecraft_mod_ai.authored_state_lowering import prepare_state_concern
    from minecraft_mod_ai.structured_state_runtime import render_state_model_concern

    task = {"implementation_obligations": [json.dumps({
        "instruction": json.dumps({"concern": concern}), "structured_records": [record],
    })]}
    original = deepcopy(task)
    with pytest.raises(ValueError):
        render_state_model_concern(task, concern, include_runtime=True)
    source, contract = prepare_state_concern(task, concern, include_runtime=True)
    assert len(contract["work"]) == 1
    work = contract["work"][0]
    assert work["record"] == record
    assert work["field"] == field
    assert work["symbol"] + "(context)" in source
    assert "public static synchronized" in source
    assert task == original


def test_state_helper_omission_is_rejected_before_compile(tmp_path):
    from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError

    task, section, active = bound_state(state_request())
    def no_compile(_):
        pytest.fail("a missing state action must not reach compilation")
    executor = AtomicConcernExecutor(
        root=tmp_path, target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="// MMM_AUTHORED_FEATURE_BODY", task=task, section=section,
        concerns=active, grounding={}, dependency_source="", require_initialize=False,
        call_coder=lambda _: 'private static final String DESCRIPTION = "trade";',
        compile_java=no_compile, compile_log=lambda _: "", write_source=lambda *_: None,
        region_attempt_limit=1,
    )
    with pytest.raises(CustomModuleGenerationError, match="missing authored state implementation"):
        executor.run()
