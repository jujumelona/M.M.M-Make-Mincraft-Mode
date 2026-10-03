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
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
from minecraft_mod_ai.implementation_graph_execution import (
    _bind_atomic_leaf_contract,
    _normalize_implementation_graph_request,
)
from minecraft_mod_ai.planning_detail_template import validate_worksheet_section


def state_request():
    section = {"specification": {
        "variables": [
            {"name": "player_currency", "owner": "Player", "type": "integer",
             "unit": "credits", "default": "100", "domain": "0 to infinity"},
            {"name": "ship_blueprint", "owner": "Player", "type": "item",
             "unit": "blueprint", "default": "present", "domain": "nullable object"},
        ],
        "transitions": [{
            "from_state": "resource_farming_idle",
            "trigger": "player_interact_with_trade_station",
            "guard": "player_currency >= trade_cost && ship_blueprint != null",
            "mutation": "player_currency -= trade_cost",
            "to_state": "currency_acquired",
        }],
        "invariants": [{
            "condition": "player_currency >= 0",
            "enforcement": "reject negative balances",
        }],
        "initialization": [],
        "updates": [],
        "cleanup": [],
        "concurrency": [],
        "inapplicable_concerns": [],
    }, "constraint_evidence_refs": []}
    structured = {"state_model": section}
    return {
        "text": "## state_model\n- variables: player_currency, ship_blueprint\n",
        "structured_sections": structured,
        "structured_sections_sha256": structured_sections_sha256(structured),
        "canonical_concern_authority": CanonicalConcernAuthority.from_structured_sections(
            structured
        ).to_dict(),
    }


def bound_state(request):
    normalized = _normalize_implementation_graph_request(request)
    task = {"task_id": "ir_authoredstatemodel"}
    section, active = _bind_atomic_leaf_contract(
        task,
        {"symbol": "AuthoredStateModel", "obligations": []},
        {},
        structured_sections=normalized["structured_sections"],
        canonical_concern_authority=normalized["canonical_concern_authority"],
        production_state_section=normalized["production_state_section"],
    )
    return task, section, active


def test_planning_boundary_accepts_only_host_compilable_state_dsl():
    request = state_request()
    original = deepcopy(request)
    section = request["structured_sections"]["state_model"]
    assert validate_worksheet_section(section, set(), "state_model") == section

    invalid = deepcopy(section)
    invalid["specification"]["transitions"][0]["mutation"] = (
        "update_player_currency(player_currency - trade_cost)"
    )
    with pytest.raises(ValueError, match="fixed specification template|host-compiled state DSL"):
        validate_worksheet_section(invalid, set(), "state_model")
    assert request == original


def test_graph_boundary_preserves_host_dsl_without_reinterpreting_it():
    request = state_request()
    original = deepcopy(request)
    normalized = _normalize_implementation_graph_request(request)
    state = normalized["production_state_section"]["specification"]
    assert state["transitions"][0]["mutation"] == "player_currency -= trade_cost"
    assert state["invariants"][0]["condition"] == "player_currency >= 0"
    assert request == original


def test_authored_state_transition_is_host_compiled_and_changes_balance(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")

    request = state_request()
    original = deepcopy(request)
    task, section, active = bound_state(request)

    def forbidden_coder(_messages):
        pytest.fail("state_model must never enter the coder")

    def compile_java(root):
        result = subprocess.run(
            [javac, "AuthoredStateModel.java", "Probe.java"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return SimpleNamespace(
            status="PASS" if result.returncode == 0 else "FAIL",
            error=result.stdout + result.stderr,
        )

    (tmp_path / "Probe.java").write_text(
        """
public class Probe {
    public static void main(String[] args) {
        java.util.Map<String, Object> context = new java.util.HashMap<>();
        context.put("trade_cost", 25);
        String next = AuthoredStateModel.transition(
                "resource_farming_idle",
                "player_interact_with_trade_station",
                context);
        if (!next.equals("currency_acquired")) throw new AssertionError(next);
        if (((Number) AuthoredStateModel.getState("player_currency")).doubleValue() != 75)
            throw new AssertionError("currency mutation was dropped");
        if (!AuthoredStateModel.invariantsHold(context))
            throw new AssertionError("invariant degraded");
        AuthoredStateModel.setState("player_currency", -1, context);
        if (AuthoredStateModel.invariantsHold(context))
            throw new AssertionError("negative balance was accepted");
        System.out.println("balance=75; negative balance rejected");
    }
}
""",
        encoding="utf-8",
    )

    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java",
        symbol="AuthoredStateModel",
        original="public final class AuthoredStateModel { // MMM_AUTHORED_FEATURE_BODY\n}",
        task=task,
        section=section,
        concerns=active,
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=forbidden_coder,
        compile_java=compile_java,
        compile_log=lambda result: result.error,
        write_source=lambda path, source: path.write_text(source, encoding="utf-8"),
    )
    result = executor.run()
    assert result["repair_count"] == 0
    assert set(executor.host_owned_concerns) == {"variables", "transitions", "invariants"}

    run = subprocess.run(
        [java, "-cp", str(tmp_path), "Probe"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "balance=75; negative balance rejected" in run.stdout
    assert request == original


def test_invalid_state_record_fails_before_coder_or_compile(tmp_path):
    request = state_request()
    request["structured_sections"]["state_model"]["specification"]["transitions"][0][
        "mutation"
    ] = "update_player_currency(player_currency - trade_cost)"
    request["structured_sections_sha256"] = structured_sections_sha256(
        request["structured_sections"]
    )
    request["canonical_concern_authority"] = CanonicalConcernAuthority.from_structured_sections(
        request["structured_sections"]
    ).to_dict()
    task, section, active = bound_state(request)

    def forbidden_coder(_messages):
        pytest.fail("invalid state must fail at the host contract, not enter coder")

    def forbidden_compile(_root):
        pytest.fail("invalid state must fail before javac/Gradle")

    executor = AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java",
        symbol="AuthoredStateModel",
        original="// MMM_AUTHORED_FEATURE_BODY",
        task=task,
        section=section,
        concerns=active,
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=forbidden_coder,
        compile_java=forbidden_compile,
        compile_log=lambda _report: "",
        write_source=lambda *_args: None,
    )
    with pytest.raises(CustomModuleGenerationError, match="STRUCTURED_STATE_HOST_DSL_REQUIRED"):
        executor.run()


def test_host_only_state_section_cannot_enter_atomic_java_coder():
    class Router:
        def generate_tool_decision(self, *_args, **_kwargs):
            pytest.fail("host-only section reached model callback")

    messages = [{
        "role": "user",
        "content": json.dumps({
            "section": "state_model",
            "response_region": "members",
            "host_selected_class": "AuthoredStateModel",
            "concern": {"name": "transitions"},
        }),
    }]
    with pytest.raises(
        CustomModuleGenerationError,
        match="ATOMIC_HOST_ONLY_SECTION_CODER_FORBIDDEN",
    ):
        _call_atomic_java_region(Router(), messages, output_token_ceiling=256)
