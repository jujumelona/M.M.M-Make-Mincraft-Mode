from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai import research_code_context
from minecraft_mod_ai import research_coder_repair_reuse as reuse


def _unit(path: str, package: str, imports=(), types=()):
    return SimpleNamespace(
        path=path,
        package=package,
        imports=tuple(imports),
        types=tuple(types),
        methods=(),
    )


def test_repocoder_has_no_legacy_fixed_round_override(monkeypatch) -> None:
    assert not hasattr(reuse, "_round_budget")
    assert not hasattr(reuse, "_bounded_evolution_state_budget")
    monkeypatch.setenv("MMM_CODE_RESEARCH_EVOLUTION_STATES", "8")
    assert custom_search._evolution_state_budget() == 8


def test_dependency_query_prioritizes_direct_reverse_and_build_neighbors() -> None:
    target = "src/main/java/demo/Target.java"
    helper = "src/main/java/demo/Helper.java"
    caller = "src/main/java/demo/Caller.java"
    contract = "src/main/java/demo/TargetContract.java"
    context = SimpleNamespace(
        byte_budget=16 * 1024,
        units={
            target: _unit(target, "demo", imports=("demo.Helper",), types=("Target",)),
            helper: _unit(helper, "demo", types=("Helper",)),
            caller: _unit(caller, "demo", imports=("demo.Target",), types=("Caller",)),
            contract: _unit(contract, "demo", types=("TargetContract",)),
        },
        index=SimpleNamespace(
            files=[SimpleNamespace(path="build.gradle"), SimpleNamespace(path=target)]
        ),
    )
    query = reuse._dependency_neighborhood_query(
        research_code_context,
        context,
        "Target dependency API",
        None,
    )
    assert target in query
    assert helper in query
    assert caller in query
    assert contract in query
    assert "build.gradle" in query


def test_diagnostic_signature_keeps_file_line_symbol_task_and_exception(tmp_path: Path) -> None:
    log = tmp_path / "build.log"
    log.write_text(
        "src/main/java/demo/Broken.java:42: error: cannot find symbol RegistryKey\n"
        "java.lang.IllegalStateException: boom\n",
        encoding="utf-8",
    )
    evidence = {
        "diagnostics": {
            "diagnostics": [
                {
                    "path": "src/main/java/demo/Broken.java",
                    "message": "cannot find symbol RegistryKey",
                    "range": {"start": {"line": 41}},
                }
            ]
        },
        "build": {
            "status": "FAIL",
            "commands": [{"task": ":gameTest", "log_path": str(log)}],
        },
    }
    signature = reuse._diagnostic_signature_payload(evidence)
    assert "src/main/java/demo/Broken.java" in signature["files"]
    assert "src/main/java/demo/Broken.java:42" in signature["lines"]
    assert "RegistryKey" in signature["symbols"]
    assert ":gameTest" in signature["tasks"]
    assert "java.lang.IllegalStateException" in signature["exceptions"]


def test_same_diagnostic_memoizes_and_new_diagnostic_augments_receipt(tmp_path: Path) -> None:
    class FakeRepairEngine:
        calls = 0

        @staticmethod
        def _signature(evidence):
            return json.dumps(evidence, sort_keys=True)

        def _context(self, root, evidence):
            type(self).calls += 1
            return {
                "manifest": {"sha256": "manifest-a"},
                "relevant": {"files": [{"path": "src/main/java/demo/Broken.java"}]},
            }

    reuse._install_repair_context_reuse(
        SimpleNamespace(RepairEngine=FakeRepairEngine)
    )
    receipt = {
        "bundle_sha256": "sha256:bundle-a",
        "reusable_evidence": [
            {
                "evidence_id": "repo:broken",
                "source_type": "repository_file",
                "path": "src/main/java/demo/Broken.java",
                "sha256": "sha256:file-a",
                "symbols": ["Broken", "RegistryKey"],
                "snippet": "class Broken { RegistryKey key; }",
            }
        ],
    }
    reuse._persist_research_receipt(tmp_path, receipt, module_id="broken-module")
    engine = FakeRepairEngine()
    first_evidence = {
        "diagnostics": {
            "diagnostics": [
                {
                    "path": "src/main/java/demo/Broken.java",
                    "message": "cannot find symbol RegistryKey",
                    "range": {"start": {"line": 4}},
                }
            ]
        },
        "build": {"status": "FAIL", "commands": [{"task": ":compileJava"}]},
    }
    first = engine._context(tmp_path, first_evidence)
    repeated = engine._context(tmp_path, first_evidence)
    assert FakeRepairEngine.calls == 1
    assert repeated == first
    assert first["prior_research_evidence"][0]["evidence_id"] == "repo:broken"
    assert first["repair_evidence_receipt"]["novel_diagnostic"] is True

    second_evidence = {
        "diagnostics": {
            "diagnostics": [
                {
                    "path": "src/main/java/demo/Broken.java",
                    "message": "no suitable method register",
                    "range": {"start": {"line": 8}},
                }
            ]
        },
        "build": {"status": "FAIL", "commands": [{"task": ":gameTest"}]},
    }
    second = engine._context(tmp_path, second_evidence)
    assert FakeRepairEngine.calls == 2
    assert second["repair_evidence_receipt"]["previous_diagnostic_receipt_sha256"]
    assert (
        second["repair_evidence_receipt"]["receipt_sha256"]
        != first["repair_evidence_receipt"]["receipt_sha256"]
    )


def test_receipt_store_dedupes_same_plan_coder_receipt(tmp_path: Path) -> None:
    receipt = {
        "bundle_sha256": "sha256:bundle-a",
        "reusable_evidence": [],
    }
    reuse._persist_research_receipt(tmp_path, receipt, module_id="module-a")
    reuse._persist_research_receipt(tmp_path, receipt, module_id="module-a")
    stored = json.loads(
        (tmp_path / ".minecraft_ai/research-code-context-receipts.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(stored["receipts"]) == 1
