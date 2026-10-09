from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import minecraft_mod_ai.complete_orchestrator as complete_orchestrator_module
from minecraft_mod_ai.colab_run_modes import write_debug_example_plan
from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_spec import CompleteProposal
from minecraft_mod_ai import platform_live_execution_contract as live_platform
from minecraft_mod_ai.platform_catalog import adapter_for_lock_values, provider_for_loader


def _assert_official_scaffold_route(proposal: CompleteProposal) -> None:
    adapter = adapter_for_lock_values(proposal.base_proposal.spec.platform)
    provider = provider_for_loader(adapter.loader)
    assert adapter.loader == "fabric"
    assert tuple(adapter.deterministic_module_kinds) == ()
    assert provider.host_authoritative is True, (
        provider.provider_id,
        provider.host_authoritative,
    )
    assert live_platform._uses_official_scaffold(adapter) is True


class _DeterministicRouter:
    profile = "deterministic-e2e"
    registry = None

    def __init__(self) -> None:
        self.workspace_root: Path | None = None

    def bind_agent_workspace(
        self,
        workspace_root: str | Path,
        *,
        require_fresh_evidence: bool = False,
    ) -> "_DeterministicRouter":
        del require_fresh_evidence
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        return self

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        del kwargs
        assert role == "coder"
        prompt = str(messages[-1]["content"])
        grounding_marker = "Host implementation grounding:\n"
        scaffold_marker = "\n\nCurrent host scaffold:"
        assert grounding_marker in prompt
        grounding = json.loads(
            prompt.split(grounding_marker, 1)[1].split(scaffold_marker, 1)[0]
        )
        assert grounding["artifact_kind"] == "item"
        fact = grounding["facts"][0]
        imports = list(dict.fromkeys(fact["required_imports"]))
        templates = list(fact["templates"])
        key_template = next(
            item
            for item in templates
            if "resource_key_create" in item.get("symbol_usage", ())
        )
        register_template = next(
            item
            for item in templates
            if "register_item" in item.get("symbol_usage", ())
        )

        key_body = key_template["render_body"]
        register_body = register_template["render_body"]
        replacements = {
            "{{java_constant}}": "DEBUG_TOKEN",
            "{{mod_id}}": "mmm_debug_fixture",
            "{{registry_path}}": "debug_token",
            "ModItemIds.DEBUG_TOKEN_KEY": "DEBUG_TOKEN_KEY",
        }
        for before, after in replacements.items():
            key_body = key_body.replace(before, after)
            register_body = register_body.replace(before, after)

        source = "\n".join(
            [
                "package dev.mmm.debugfixture;",
                "",
                *(f"import {owner};" for owner in imports),
                "",
                "public final class DebugToken {",
                "    private DebugToken() {}",
                "",
                *(
                    "    " + line if line else ""
                    for line in key_body.strip().splitlines()
                ),
                "",
                *(
                    "    " + line if line else ""
                    for line in register_body.strip().splitlines()
                ),
                "}",
                "",
            ]
        )
        return source


def test_debug_fixture_runs_real_build_and_packaging_without_live_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    proposal = CompleteProposal.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    _assert_official_scaffold_route(proposal)
    orchestrator = CompleteProductionOrchestrator(
        workspace_root=tmp_path / "workspace",
        profile="deterministic-e2e",
        router_factory=_DeterministicRouter,
    )
    _assert_official_scaffold_route(proposal)
    _original_official_scaffold_route = live_platform._uses_official_scaffold

    def _checked_official_scaffold_route(adapter):
        provider = provider_for_loader(adapter.loader)
        result = _original_official_scaffold_route(adapter)
        assert result is True, {
            "loader": adapter.loader,
            "adapter_id": adapter.adapter_id,
            "source_api_family": adapter.source_api_family,
            "deterministic_module_kinds": list(adapter.deterministic_module_kinds),
            "provider_id": provider.provider_id,
            "provider_host_authoritative": provider.host_authoritative,
        }
        return result

    monkeypatch.setattr(
        live_platform,
        "_uses_official_scaffold",
        _checked_official_scaffold_route,
    )
    result = orchestrator.execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="debug-full-pipeline-e2e",
        options=CompleteExecutionOptions(
            source_only=False,
            run_jdt=True,
            run_gametest=True,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
            cleanup_runtime=True,
            eula_accepted=True,
        ),
    )

    project_root = Path(result.project_root)
    target = project_root / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    assert target.is_file()
    assert result.build_report is not None
    assert result.build_report["status"] == "PASS"
    assert CompleteProductionOrchestrator._full_gradle_build_receipt_passed(
        result.build_report
    )
    assert result.jar_validation is not None
    assert result.jar_validation["status"] == "PASS"
    jdt_receipts = [
        receipt
        for receipt in result.module_receipts
        if receipt.get("schema_version") == "mmm/java-diagnostics-v3"
        and receipt.get("verification_backend") == "jdt_core"
    ]
    assert len(jdt_receipts) == 1
    assert jdt_receipts[0]["complete"] is True
    assert int(jdt_receipts[0]["error_count"]) == 0
    assert all(
        str(jdt_receipts[0].get(key) or "").strip()
        for key in ("model_id", "model_revision", "session_id")
    )
    assert "execution-gate:jdt:missing-jdt" not in result.unresolved_gates
    texture = (
        project_root
        / "src/main/resources/assets/mmm_debug_fixture/textures/item/debug_token.png"
    )
    assert texture.is_file()
    assert texture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    en_us = json.loads(
        (
            project_root
            / "src/main/resources/assets/mmm_debug_fixture/lang/en_us.json"
        ).read_text(encoding="utf-8")
    )
    ko_kr = json.loads(
        (
            project_root
            / "src/main/resources/assets/mmm_debug_fixture/lang/ko_kr.json"
        ).read_text(encoding="utf-8")
    )
    assert en_us["item.mmm_debug_fixture.debug_token"] == "Debug Token"
    assert ko_kr["item.mmm_debug_fixture.debug_token"] == "디버그 토큰"

    with __import__("zipfile").ZipFile(result.jar_path, "r") as jar:
        jar_names = set(jar.namelist())
        assert "assets/mmm_debug_fixture/textures/item/debug_token.png" in jar_names
        assert "assets/mmm_debug_fixture/lang/en_us.json" in jar_names
        assert "assets/mmm_debug_fixture/lang/ko_kr.json" in jar_names
        assert any(
            name.startswith("assets/mmm_debug_fixture/")
            and name.endswith("/debug_token.json")
            for name in jar_names
        )
    assert result.build_bundle_zip is not None
    assert Path(result.build_bundle_zip).is_file()
    assert result.release_zip is not None
    assert Path(result.release_zip).is_file()
    with __import__("zipfile").ZipFile(result.release_zip, "r") as archive:
        manifest = json.loads(archive.read("release-manifest.json"))
        receipt_rows = [
            json.loads(line)
            for line in archive.read(
                "source/.minecraft_ai/production-receipts.jsonl"
            ).decode("utf-8").splitlines()
            if line.strip()
        ]
    assert manifest["proposal_hash"] == proposal.calculate_hash()
    assert manifest["base_proposal_hash"] == proposal.base_proposal.calculate_hash()
    assert manifest["proposal_scope"] == "complete"
    summary = next(row["value"] for row in receipt_rows if row["record_type"] == "summary")
    task_states = {
        row["node_id"]: row["state"]
        for row in receipt_rows
        if row["record_type"] == "task"
    }
    assert summary["task_count"] >= 1
    assert summary["counts"].get("failed", 0) == 0
    assert task_states["package-build-artifact"] == "succeeded"
    assert task_states["package-release"] == "pending"
    assert not any(
        gate.startswith("required-gate:")
        for gate in result.unresolved_gates
    )


def test_debug_fixture_releases_when_only_jdt_infrastructure_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        complete_orchestrator_module,
        "_run_release_jdt_verification",
        lambda _project_root: {
            "status": "UNAVAILABLE",
            "error": (
                "JDTWorkspaceBootstrapError: language/status ServiceReady "
                "was not observed before validation"
            ),
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 0,
            "verification_attempts": 2,
        },
    )
    plan_path = write_debug_example_plan(
        tmp_path / "proposal.json",
        minecraft_version="1.21.8",
        loader="fabric",
    )
    proposal = CompleteProposal.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    orchestrator = CompleteProductionOrchestrator(
        workspace_root=tmp_path / "workspace",
        profile="deterministic-e2e",
        router_factory=_DeterministicRouter,
    )

    result = orchestrator.execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="debug-jdt-unavailable-build-bundle",
        options=CompleteExecutionOptions(
            source_only=False,
            run_jdt=True,
            run_gametest=True,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
            cleanup_runtime=True,
            eula_accepted=True,
        ),
    )

    assert result.status == "VERIFIED"
    assert result.release_ready is True
    assert result.release_zip is not None
    assert Path(result.release_zip).is_file()
    assert result.jar_path is not None
    assert Path(result.jar_path).is_file()
    assert result.build_bundle_zip is not None
    assert Path(result.build_bundle_zip).is_file()
    assert "execution-gate:jdt:missing-jdt" not in result.unresolved_gates

    with __import__("zipfile").ZipFile(result.build_bundle_zip, "r") as archive:
        manifest = json.loads(archive.read("build-manifest.json"))
        names = set(archive.namelist())
        jdt_receipt = json.loads(archive.read("receipts/jdt-receipt.json"))
    # The build artifact bundle is never itself a certified release; the
    # separately packaged release ZIP carries release certification.
    assert manifest["release_certified"] is False
    assert manifest["release_ready"] is True
    assert "execution-gate:jdt:missing-jdt" not in manifest["unresolved_gates"]
    assert jdt_receipt["status"] == "UNAVAILABLE"
    assert any(name.startswith("artifact/") and name.endswith(".jar") for name in names)
