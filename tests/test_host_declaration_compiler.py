from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
from minecraft_mod_ai.authored_atomic_contract import build_authored_atomic_contract
from minecraft_mod_ai.authored_execution_schema import concern_contracts
from minecraft_mod_ai.canonical_concern_authority import CanonicalConcernAuthority
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import _call_atomic_java_region
from minecraft_mod_ai.execution_contract_policy import (
    JAVA_DECLARATION_ONLY_CONCERNS,
    java_atomic_parameters_for_request,
)


def forbidden_coder(*args, **kwargs):
    raise AssertionError("coder must not be called")


def declaration_executor(tmp_path, records, *, symbol="AuthoredPersistence"):
    authority = CanonicalConcernAuthority({"persistence": {"stored_state": records}})
    contract = build_authored_atomic_contract(
        section="persistence",
        concerns=concern_contracts("persistence"),
        requirements={},
        raw_obligations=[],
        canonical_concern_authority=authority,
    )
    return AtomicConcernExecutor(
        root=tmp_path,
        target=tmp_path / f"{symbol}.java",
        relative=f"{symbol}.java",
        symbol=symbol,
        original="// MMM_AUTHORED_FEATURE_BODY\n",
        task={"authored_atomic_contract": contract},
        section="persistence",
        concerns=contract["active_concerns"],
        grounding={},
        dependency_source="",
        require_initialize=False,
        call_coder=forbidden_coder,
        completion_decider=forbidden_coder,
        compile_java=lambda _: SimpleNamespace(status="PASS"),
        compile_log=lambda report: report.error,
        write_source=lambda path, source: path.write_text(source, encoding="utf-8"),
        compile_repair_limit=3,
    )


@pytest.mark.parametrize("symbol", ["AuthoredPersistence", "StoredState"])
def test_stored_state_compiles_and_runs_without_any_model(tmp_path, symbol):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK required")
    rows = [
        {"state": "credits", "owner": "player", "scope": "world"},
        {"state": 'quote"\\newline\n한글', "owner": "StoredState", "scope": "server"},
        *({"state": f"v{i}", "owner": "player", "scope": "world"} for i in range(12)),
    ]
    executor = declaration_executor(tmp_path, rows, symbol=symbol)

    def compile_java(root):
        compiled = subprocess.run(
            [javac, "-encoding", "UTF-8", str(executor.target)],
            cwd=root, capture_output=True, text=True, timeout=30, check=False,
        )
        return SimpleNamespace(
            status="PASS" if compiled.returncode == 0 else "FAIL",
            error=compiled.stderr,
        )

    executor.compile_java = compile_java
    result = executor.run()
    assert result["repair_count"] == 0
    assert executor.host_owned_concerns == {"stored_state"}
    # Exercise the generated declarations, including literal round-tripping and
    # every row beyond the old small-model page limit, in a real JVM.
    expected = json.dumps(rows[1]["state"], ensure_ascii=True)
    probe = f'''
import java.lang.reflect.*;
import java.util.*;
public class Probe {{
    public static void main(String[] args) throws Exception {{
        Field f = {symbol}.class.getDeclaredField("STORED_STATES");
        f.setAccessible(true);
        List<?> values = (List<?>) f.get(null);
        if (values.size() != 14) throw new AssertionError(values.size());
        Object value = values.get(1);
        if (!value.getClass().isRecord()) throw new AssertionError("not record");
        for (String component : new String[] {{"state", "owner", "scope"}}) {{
            Method accessor = value.getClass().getDeclaredMethod(component);
            accessor.setAccessible(true);
            String expected = switch (component) {{
                case "state" -> {expected};
                case "owner" -> "StoredState";
                default -> "server";
            }};
            if (!expected.equals(accessor.invoke(value))) throw new AssertionError(component);
        }}
        try {{ values.clear(); throw new AssertionError("mutable"); }}
        catch (UnsupportedOperationException expectedException) {{}}
        System.out.println("PASS");
    }}
}}
'''
    (tmp_path / "Probe.java").write_text(probe, encoding="utf-8")
    compiled = subprocess.run(
        [javac, "-encoding", "UTF-8", "Probe.java"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "Probe"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout.strip() == "PASS"


@pytest.mark.parametrize("records", [[], [{"state": "credits", "owner": "player"}],
    [{"state": 42, "owner": "player", "scope": "world"}]])
def test_missing_or_invalid_records_fail_before_coder_and_write(tmp_path, records):
    executor = declaration_executor(tmp_path, records)
    with pytest.raises(CustomModuleGenerationError, match="HOST_DECLARATION_RECORDS_INVALID"):
        executor.run()
    assert not executor.target.exists()


@pytest.mark.parametrize("name", sorted(JAVA_DECLARATION_ONLY_CONCERNS))
@pytest.mark.parametrize("region", ["members", "initialize"])
def test_declaration_only_has_no_model_schema_or_coder_entry(name, region):
    payload = {"section": "persistence", "concern": {"name": name},
               "response_region": region, "host_selected_class": "AuthoredPersistence"}
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN"):
        java_atomic_parameters_for_request(payload, response_region=region)
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN"):
        _call_atomic_java_region(
            SimpleNamespace(generate_tool_decision=forbidden_coder),
            [{"role": "user", "content": json.dumps(payload)}],
            output_token_ceiling=128,
        )


def test_direct_assembly_cannot_bypass_host_declaration_ownership():
    from minecraft_mod_ai.atomic_java_assembly import JavaStructureAssembly
    from minecraft_mod_ai.execution_contract_policy import (
        JAVA_ATOMIC_MEMBERS_PARAMETERS,
    )

    assembly = JavaStructureAssembly(
        forbidden_coder, {"concern": {"name": "stored_state"}},
        output_token_ceiling=128,
    )
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN"):
        assembly.run(JAVA_ATOMIC_MEMBERS_PARAMETERS)


def test_host_declarations_do_not_reenter_coder_on_compile_failure(tmp_path):
    executor = declaration_executor(tmp_path, [{"state": "credits", "owner": "player", "scope": "world"}])

    def fail_compile(_):
        line = next(i for i, value in enumerate(executor.source.splitlines(), 1) if "record " in value)
        return SimpleNamespace(status="FAIL", error=f"AuthoredPersistence.java:{line}: error: deliberate compiler failure")

    executor.compile_java = fail_compile
    with pytest.raises(CustomModuleGenerationError, match="HOST_COMPILER_INVALID"):
        executor.run()


@pytest.mark.parametrize("invalid", [42, None, "not a record", []])
def test_invalid_record_cannot_be_silently_dropped(tmp_path, invalid):
    valid = {"state": "credits", "owner": "player", "scope": "world"}
    executor = declaration_executor(tmp_path, [valid])
    executor.task["authored_atomic_contract"]["concerns"]["stored_state"]["structured_records"] = [valid, invalid]
    with pytest.raises(CustomModuleGenerationError, match="HOST_DECLARATION_RECORDS_INVALID"):
        executor.run()
    assert not executor.target.exists()


def test_declaration_only_cannot_enter_paging_or_completion():
    from minecraft_mod_ai.atomic_region_paging import (
        decide_region_completion,
        generate_region,
    )

    payload = {"section": "persistence", "concern": {"name": "stored_state"}, "response_region": "members"}
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN"):
        generate_region(
            forbidden_coder, [{"role": "user", "content": json.dumps(payload)}],
            completion_decider=forbidden_coder,
        )
    with pytest.raises(CustomModuleGenerationError, match="ATOMIC_HOST_ONLY_CONCERN_CODER_FORBIDDEN"):
        decide_region_completion(SimpleNamespace(generate_tool_decision=forbidden_coder), payload)
