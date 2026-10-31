"""State ABI belongs to the host, including value types and declared names."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_production_state_compiler import _state_section

from minecraft_mod_ai.construction_source_repair import assemble_java_implementation
from minecraft_mod_ai.host_prerequisite_api import java_api_declarations
from minecraft_mod_ai.production_state_compiler import render_production_state_java


def _source():
    variables = [
        {"name": name, "type": kind, "default": default, "owner": "server", "unit": "value", "domain": "test"}
        for name, kind, default in [
            ("credits", "long", "9"), ("count", "int", "7"), ("ready", "boolean", "true"),
            ("factor", "double", "2.5"), ("label", "string", "ready"),
        ]
    ]
    return render_production_state_java(_state_section(variables=variables), package_name="proof")


def test_host_generated_state_accessors_execute_with_exact_value_types(tmp_path):
    javac = shutil.which("javac")
    assert javac, "This regression requires an actual JDK compiler"
    state = tmp_path / "proof/AuthoredStateModel.java"
    state.parent.mkdir()
    state.write_text(_source(), encoding="utf-8")
    probe = state.with_name("StateProbe.java")
    probe.write_text('''package proof;
public class StateProbe {
    public static void main(String[] args) {
        long credits = AuthoredStateModel.readState_credits();
        int count = AuthoredStateModel.readState_count();
        boolean ready = AuthoredStateModel.readState_ready();
        double factor = AuthoredStateModel.readState_factor();
        String label = AuthoredStateModel.readState_label();
        if (credits != 9L || count != 7 || !ready || factor != 2.5d || !"ready".equals(label)) throw new AssertionError("defaults");
        AuthoredStateModel.writeState_credits(25L);
        java.util.Map<String, Object> context = new java.util.HashMap<>();
        context.put("credits", 41L);
        if (AuthoredStateModel.readState_credits(context) != 41L) throw new AssertionError("read overlay");
        if (AuthoredStateModel.readState_credits() != 25L) throw new AssertionError("read mutated state");
        AuthoredStateModel.writeState_credits(53L, context);
        if (AuthoredStateModel.readState_credits(context) != 53L || AuthoredStateModel.readState_credits() != 53L) throw new AssertionError("write context");
        AuthoredStateModel.setState("count", null);
        if (AuthoredStateModel.readState_count() != 7) throw new AssertionError("declared default");
        System.out.println("STATE_ACCESS_PASS");
    }
}
''', encoding="utf-8")
    compiled = subprocess.run([javac, "-proc:none", "-d", str(tmp_path), str(state), str(probe)], capture_output=True, text=True, timeout=30, check=False)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(Path(javac).with_name('java')), "-cp", str(tmp_path), "proof.StateProbe"], capture_output=True, text=True, timeout=30, check=False)
    assert executed.returncode == 0, executed.stderr
    assert "STATE_ACCESS_PASS" in executed.stdout


def _frame():
    from minecraft_mod_ai.host_typed_state_access import state_access_contract
    apis = [java_api_declarations(_source(), path="src/main/java/proof/AuthoredStateModel.java")]
    return {"kind": "method", "prefix": "private String handle() {\n", "suffix": "\n}",
            "host_state_access": state_access_contract(apis)}


def test_production_get_state_list_argument_is_rejected_before_compiler():
    # Exact failure shape from production-logs (19), not a fictitious new API.
    body = 'String selectedSection = AuthoredStateModel.getState("selected_section", java.util.List.of(player.getUUID().toString())); return selectedSection;'
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS") as caught:
        assemble_java_implementation(json.dumps({"implementation": body}), _frame())
    assert "readState_label" in str(caught.value)
    assert "selected_section" in str(caught.value)
    assert caught.value.failed_source


@pytest.mark.parametrize("call", ['clearScopedState()', 'exportState("label")', 'importState("label", "replacement")', 'defaultStateSnapshot("label", "replacement")'])
def test_persistence_lifecycle_cannot_be_called_by_a_generated_behavior(call):
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS"):
        assemble_java_implementation(json.dumps({"implementation": f"AuthoredStateModel.{call}; return null;"}), _frame())
def test_only_declared_typed_state_names_are_admitted():
    with pytest.raises(ValueError, match="JAVA_HOST_STATE_ACCESS_UNKNOWN"):
        assemble_java_implementation(json.dumps({"implementation": "return AuthoredStateModel.readState_selected_section();"}), _frame())
    body = 'String text = "AuthoredStateModel.getState(\\\"unused\\\")"; /* AuthoredStateModel.setState("x", null); */ return proof.AuthoredStateModel.readState_label();'
    assert "return proof.AuthoredStateModel.readState_label();" in assemble_java_implementation(json.dumps({"implementation": body}), _frame())


def test_host_prompt_exposes_typed_state_api_without_its_forbidden_raw_bridge(tmp_path, monkeypatch):
    from copy import deepcopy

    from test_construction_host_source import SPEC, _context

    from minecraft_mod_ai.construction_host_source import _build_host_java_prompt

    relative = "src/main/java/proof/AuthoredStateModel.java"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(_source())
    evidence = {
        "target_api": [{"kind": "METHOD", "owner": "proof.AuthoredStateModel",
                        "name": "getState", "parameters": ["String"], "return_type": "Object"}],
        "source": [{"path": relative, "code": _source()}],
        "class_members": [{"verified_class": "proof.AuthoredStateModel", "members": [
            {"kind": "METHOD", "owner": "proof.AuthoredStateModel", "name": "getState"},
            {"kind": "METHOD", "owner": "proof.AuthoredStateModel", "name": "readState_credits"},
        ]}],
    }
    original = deepcopy(evidence)
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence", lambda *a, **k: evidence)
    _, _, _, prompt, access = _build_host_java_prompt(
        None, SPEC, _context(tmp_path), package="example", name="Bridge", owner="example.Bridge",
        behaviors=[], source_context=[{"path": relative}],
    )
    exposed = json.dumps(prompt)
    assert "getState" not in exposed
    assert "setState" not in exposed
    assert "readState_credits" in exposed
    assert "writeState_credits" in exposed
    assert access[0]["owner"] == "proof.AuthoredStateModel"
    assert evidence == original


def test_method_references_cannot_bypass_state_contract():
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS"):
        assemble_java_implementation(json.dumps({"implementation": "java.util.function.Function<String, Object> read = AuthoredStateModel::getState; return null;"}), _frame())
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS"):
        assemble_java_implementation(json.dumps({"implementation": 'return proof /* qualifier */ . AuthoredStateModel.getState("label");'}), _frame())


def test_raw_state_failure_uses_existing_native_correction_budget(monkeypatch):
    from types import SimpleNamespace

    from test_first_pass_java_data_transport import DataRouter

    from minecraft_mod_ai.model_adapters.base import AdapterConfig

    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context", lambda messages, **k: messages)
    requests = []

    def generate(request):
        requests.append(request)
        body = ('return AuthoredStateModel.getState("label", java.util.List.of("invented-context"));'
                if len(requests) == 1 else 'return AuthoredStateModel.readState_label();')
        return json.dumps({"implementation": body})

    adapter = SimpleNamespace(config=AdapterConfig(role="coder", adapter="llama_cpp"), generate=generate)
    result = DataRouter(adapter).generate_java_source("coder", [], data_name="typed_state_regression", source_frame=_frame())
    assert "return AuthoredStateModel.readState_label();" in result
    assert len(requests) == 2
    correction = json.loads(requests[1].messages[-1]["content"])
    assert "JAVA_HOST_RAW_STATE_ACCESS" in correction["diagnostic"]
    assert "readState_label" in correction["diagnostic"]


def test_accessor_identity_matches_existing_runtime_normalization():
    source = render_production_state_java(_state_section(variables=[{
        "name": " credits ", "type": " long ", "default": " 9 ",
        "owner": "server", "unit": "credits", "domain": "nonnegative",
    }]), package_name="proof")
    assert "long readState_credits()" in source
    assert 'Object value = getState("credits", context);' in source
    assert "readState_ credits" not in source


@pytest.mark.parametrize("side", ["COMMON", "CLIENT"])
def test_host_state_contract_reaches_generated_members_lifecycle_and_repair(tmp_path, monkeypatch, side):
    from test_construction_host_source import SPEC, Router, _context

    from minecraft_mod_ai.construction_host_source import compose_host_java_source

    state_path = "src/main/java/proof/AuthoredStateModel.java"
    state_file = tmp_path / state_path
    state_file.parent.mkdir(parents=True)
    state_file.write_text(_source(), encoding="utf-8")
    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._evidence", lambda *a, **k: {})
    monkeypatch.setattr("minecraft_mod_ai.construction_coverage.record_authored_source_calls", lambda *a, **k: None)

    def compile_source(source, *a, **k):
        if "return missing;" in source or "missing();" in source:
            line = next(i for i, text in enumerate(source.splitlines(), 1) if "return missing;" in text or "missing();" in text)
            raise RuntimeError(f"Bridge.java:{line}: error: cannot find symbol")

    monkeypatch.setattr("minecraft_mod_ai.construction_file_author._verify_full_source", compile_source)
    calls = []

    class StateFrameRouter(Router):
        def generate_java_source(self, role, messages, *, data_name, source_frame):
            calls.append(data_name)
            assert source_frame["host_state_access"][0]["owner"] == "proof.AuthoredStateModel"
            if data_name.startswith("repair_"):
                body = 'System.out.println("repaired");' if "void " in source_frame["prefix"] else "return true;"
            elif "entrypoint" in data_name:
                body = "missing();"
            else:
                body = "return missing;"
            return source_frame["prefix"] + body + source_frame["suffix"]

    compose_host_java_source(StateFrameRouter(), {**SPEC, "side": side}, _context(tmp_path),
                             package="example", name="Bridge", owner="example.Bridge",
                             source_context=[{"path": state_path}])
    assert any(name.startswith("repair_") for name in calls)
    if side == "CLIENT":
        assert "submit_java_entrypoint_source" in calls


def test_player_state_contract_survives_declared_api_and_blocks_contextless_access():
    from minecraft_mod_ai.host_typed_state_access import (
        state_access_contract, state_prompt_api,
        validate_behavior_state_provenance,
    )
    state = render_production_state_java(_state_section(variables=[{
        "name": "balance", "owner": "player", "type": "long",
        "unit": "credits", "default": "0", "domain": "nonnegative",
    }]), package_name="proof")
    api = java_api_declarations(
        state, path="src/main/java/proof/AuthoredStateModel.java")
    contracts = state_access_contract([api])
    assert contracts[0]["state_scopes"] == {"balance": "player"}
    filtered, _ = state_prompt_api(
        [api], {"source": [], "target_api": [], "class_members": [],
                "typed_input_producers": []}, contracts)
    headers = " ".join(filtered[0]["types"][0]["members"])
    assert "mmmPlayerScoped_balance" not in headers
    assert "writeState_balance(long value)" not in headers
    assert "writeState_balance(long value, java.util.Map<String, Object> context)" in headers
    behavior = {"behavior_id": "credit", "state_reads": [], "state_writes": ["balance"]}
    with pytest.raises(ValueError, match="JAVA_PLAYER_STATE_CONTEXT_REQUIRED"):
        validate_behavior_state_provenance(
            "void run() { proof.AuthoredStateModel.writeState_balance(9L); }",
            contracts, behavior)
    validate_behavior_state_provenance(
        'void run() { proof.AuthoredStateModel.writeState_balance(9L, '
        'java.util.Map.of("player", "some-id")); }',
        contracts, behavior)
