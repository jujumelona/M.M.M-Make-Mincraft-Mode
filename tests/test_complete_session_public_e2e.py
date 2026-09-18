from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import minecraft_mod_ai.complete_build_repair as build_repair_module
import minecraft_mod_ai.complete_orchestrator as orchestrator_module
import minecraft_mod_ai.generator as generator_module
import minecraft_mod_ai.mcp_tools as mcp_tools_module
import minecraft_mod_ai.model_router as model_router_module
from minecraft_mod_ai.api import CompleteModAISession
from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_spec import (
    CompleteProposal,
    CompleteProposalStatus,
    ProductionModule,
)
from minecraft_mod_ai.knowledge import evidence_for_target
from minecraft_mod_ai.spec import (
    ContentKind,
    ContentSpec,
    ModSpec,
    PlatformLock,
    Proposal,
    ProposalStatus,
    platform_receipt_sha256,
)


class _DictReport:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


class _FakeRouter:
    def __init__(self, *, profile: str) -> None:
        self.profile = profile


class _FakeGradleRunner:
    calls = 0

    def __init__(self, _cache: Path) -> None:
        pass

    def build(self, project_root: Path, *, run_gametest: bool) -> _DictReport:
        type(self).calls += 1
        jar_path = Path(project_root) / "build" / "libs" / "public-e2e-1.0.0.jar"
        if type(self).calls == 1:
            return _DictReport(
                {
                    "status": "FAIL",
                    "jar_path": str(jar_path),
                    "commands": [
                        {"name": "build", "exit_code": 1, "timed_out": False}
                    ],
                }
            )

        jar_path.parent.mkdir(parents=True, exist_ok=True)
        fabric_mod = Path(project_root) / "src" / "main" / "resources" / "fabric.mod.json"
        processed_metadata = (
            fabric_mod.read_text(encoding="utf-8")
            .replace("${version}", "1.0.0")
            .replace("${mod_version}", "1.0.0")
        )
        with zipfile.ZipFile(jar_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("fabric.mod.json", processed_metadata)
            archive.writestr("com/example/publice2e/PublicE2eMod.class", b"e2e")
        return _DictReport(
            {
                "status": "PASS",
                "jar_path": str(jar_path),
                "commands": [
                    {"name": "build", "exit_code": 0, "timed_out": False}
                ],
                "run_gametest": run_gametest,
            }
        )


class _FakeRepairEngine:
    calls = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def repair(
        self,
        project_root: Path,
        *,
        run_gametest: bool = True,
        max_attempts: int | None = None,
    ) -> dict[str, object]:
        del max_attempts
        type(self).calls += 1
        validation = _FakeGradleRunner(Path('.')).build(
            project_root, run_gametest=run_gametest
        ).to_dict()
        assert validation.get("status") == "PASS"
        return {
            "status": "PASS",
            "attempts": 1,
            "evidence": {"passed": True, "build": validation},
            "patch_receipts": [],
        }


def _platform_lock() -> PlatformLock:
    unsealed = PlatformLock(
        edition="java",
        loader="fabric",
        minecraft_version="1.21.1",
        java_version="21",
        yarn_mappings="1.21.1+build.3",
        fabric_loader="0.16.9",
        fabric_api="0.116.1+1.21.1",
        fabric_loom="1.10-SNAPSHOT",
        gradle="8.12",
        adapter_id="public-e2e-provider",
        mappings_kind="yarn",
        mappings_version="1.21.1+build.3",
        gradle_sha256="0" * 64,
        gradle_distribution_url="https://services.gradle.org/distributions/gradle-8.12-bin.zip",
        data_pack_version="48",
        resource_pack_version="34.0",
        resource_pack_format=34,
        release_metadata_url="https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-1",
        source_api_family="fabric",
        deterministic_module_kinds=("item", "block"),
    )
    return replace(unsealed, receipt_sha256=platform_receipt_sha256(unsealed))


def _proposal() -> CompleteProposal:
    platform = _platform_lock()
    evidence = evidence_for_target(
        "Fabric project item build",
        minecraft_version=platform.minecraft_version,
    )
    base = Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=ProposalStatus.AWAITING_APPROVAL,
        requested_prompt="Add a deterministic debug token item.",
        spec=ModSpec(
            mod_id="public_e2e",
            mod_name="Public E2E",
            package_name="com.example.publice2e",
            version="1.0.0",
            summary="Public complete-session production-path fixture.",
            contents=(
                ContentSpec(
                    content_id="base_token",
                    kind=ContentKind.ITEM,
                    display_name_en="Base Token",
                    display_name_ko="기본 토큰",
                    recipe=False,
                ),
            ),
            platform=platform,
        ),
        assumptions=(),
        exclusions=(),
        deferred_requests=(),
        acceptance_tests=("The generated mod contains the requested token.",),
        evidence_sources=evidence,
    ).with_hash()
    return CompleteProposal(
        schema_version="mmm/complete-proposal-v1",
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=base.requested_prompt,
        base_proposal=base,
        game_design={"summary": "Generate one deterministic extra item."},
        modules=(
            ProductionModule(
                module_id="debug_token",
                kind="item",
                config={
                    "display_name": "Debug Token",
                    "display_name_en": "Debug Token",
                    "display_name_ko": "디버그 토큰",
                    "main_color": "#7A5CFF",
                },
                required_gates=("registry", "resource", "gradle", "jar validation"),
            ),
        ),
        acceptance_tests=("Debug Token is generated and packaged.",),
        external_runtime_required=False,
    )


def test_complete_session_build_uses_real_orchestrator_and_repair_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _FakeGradleRunner.calls = 0
    _FakeRepairEngine.calls = 0

    monkeypatch.setattr(model_router_module, "ModelRouter", _FakeRouter)
    monkeypatch.setattr(
        generator_module,
        "adapter_for_lock_values",
        lambda value: SimpleNamespace(resource_pack_format=value.resource_pack_format),
    )
    monkeypatch.setattr(build_repair_module, "GradleRunner", _FakeGradleRunner)
    monkeypatch.setattr(build_repair_module, "RepairEngine", _FakeRepairEngine)

    def fake_verify_final_mod_artifact(project_root: Path, **expected: str) -> _DictReport:
        jar = Path(project_root) / "build" / "libs" / "public-e2e-1.0.0.jar"
        digest = "sha256:" + hashlib.sha256(jar.read_bytes()).hexdigest()
        return _DictReport(
            {
                "artifact_path": str(jar.resolve()),
                "sha256": digest,
                "integrity": "PASS",
                "artifact": jar.name,
                "loader": expected["expected_loader"],
                "minecraft_version": expected["expected_minecraft_version"],
                "java": expected["expected_java"],
                "gradle": expected["expected_gradle"],
            }
        )

    monkeypatch.setattr(
        orchestrator_module,
        "verify_final_mod_artifact",
        fake_verify_final_mod_artifact,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "validate_jar",
        lambda _jar, _spec: _DictReport({"status": "PASS", "checks_run": 1}),
    )

    def fake_package_release(
        _self,
        project_root: str,
        _proposal: dict[str, object],
        _approval_hash: str,
        *,
        output_zip: str,
        jar_path: str,
    ) -> dict[str, object]:
        destination = Path(project_root).resolve().parent / output_zip
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(jar_path, Path(jar_path).name)
        return {"status": "PASS", "release_zip": str(destination.resolve())}

    monkeypatch.setattr(
        mcp_tools_module.MMMToolService,
        "package_release",
        fake_package_release,
    )

    session = CompleteModAISession(
        output_root=tmp_path / "out",
        model_profile="public-e2e",
    )
    assert type(session.orchestrator) is CompleteProductionOrchestrator
    session.complete_proposal = _proposal()

    result = session.build(
        run_name="public-entrypoint-e2e",
        options=CompleteExecutionOptions(
            source_only=False,
            run_jdt=False,
            run_gametest=False,
            auto_repair=True,
            run_blockbench=False,
            run_runtime=False,
            run_client=False,
            run_mineflayer=False,
            run_visual_review=False,
            resume=False,
        ),
    )

    assert result.status != "SOURCE_READY"
    assert result.source_validation["status"] == "PASS"
    assert result.build_report is not None
    assert result.build_report["status"] == "PASS"
    assert result.jar_path is not None and Path(result.jar_path).is_file()
    assert _FakeGradleRunner.calls == 2
    assert _FakeRepairEngine.calls == 1
    assert any(
        receipt.get("schema_version") == "mmm/repair-receipt-v2"
        for receipt in result.module_receipts
    )
    assert any(
        receipt.get("schema_version") == "mmm/extended-content-v2"
        for receipt in result.module_receipts
    )
