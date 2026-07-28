from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .broker import LocalPolicyBroker, ToolAction, approved_request
from .capabilities import capability_manifest
from .generator import FabricProjectGenerator
from .importer import ExistingProjectReport, inspect_existing_project_archive
from .orchestration import (
    make_worker_receipt,
    project_ir,
    receipt_json_line,
)
from .planner import HeuristicPlanner, Planner
from .runner import BuildReport, GradleRunner
from .spec import DeferredRequest, Proposal, ProposalStatus, SpecValidationError
from .validator import Finding, ProjectValidator, ValidationReport, validate_jar


@dataclass(frozen=True)
class PipelineResult:
    status: str
    project_root: str
    release_dir: str
    release_zip: str
    jar_path: str | None
    proposal_hash: str
    validation_status: str
    build_status: str
    gametest_status: str
    release_ready: bool
    existing_input_kind: str | None
    imported_source_snapshot_hash: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MinecraftModPipeline:
    def __init__(
        self,
        *,
        planner: Planner | None = None,
        broker: LocalPolicyBroker | None = None,
    ) -> None:
        self.planner = planner or HeuristicPlanner()
        self.broker = broker or LocalPolicyBroker()
        self.generator = FabricProjectGenerator()
        self.validator = ProjectValidator()

    def plan(
        self,
        prompt: str,
        *,
        existing_input: str | Path | None = None,
    ) -> Proposal:
        """Create an in-memory proposal. This method performs zero file writes."""
        proposal = self.planner.plan(prompt)
        if existing_input is not None:
            report = inspect_existing_project_archive(existing_input)
            proposal = self._bind_existing_input(proposal, report)
        proposal.validate()
        return proposal

    def execute(
        self,
        proposal: Proposal,
        *,
        approval_hash: str,
        output_root: Path,
        build: bool = True,
        run_gametest: bool = True,
        gradle_cache: Path | None = None,
        existing_input: str | Path | None = None,
    ) -> PipelineResult:
        approved = proposal.approve(approval_hash)
        if approved.status is not ProposalStatus.APPROVED:
            raise SpecValidationError("Proposal approval did not complete.")
        existing_report = self._verify_existing_input_binding(
            approved,
            existing_input=existing_input,
        )
        if (
            not approved.spec.contents
            and approved.spec.boss is None
            and approved.spec.arena is None
        ):
            raise SpecValidationError(
                "아직 만들기로 확정된 기능이 없습니다. 대화에서 필요한 기능을 더 정해 주세요."
            )

        output_root = output_root.resolve()
        resolved_gradle_cache = (
            gradle_cache.resolve() if gradle_cache is not None else output_root / ".cache"
        )
        try:
            resolved_gradle_cache.relative_to(output_root)
        except ValueError as exc:
            raise SpecValidationError(
                "gradle_cache must stay inside the approved output root."
            ) from exc
        if resolved_gradle_cache == output_root:
            raise SpecValidationError(
                "gradle_cache may not target the broad output root itself."
            )
        output_root.mkdir(parents=True, exist_ok=True)
        workspaces = output_root / "workspaces"
        releases = output_root / "releases"
        workspaces.mkdir(exist_ok=True)
        releases.mkdir(exist_ok=True)
        project_root = workspaces / approved.spec.mod_id
        if project_root.exists():
            raise FileExistsError(
                f"Project already exists: {project_root}. "
                "Use a new output directory or archive the existing release."
            )
        staging = workspaces / f".staging-{approved.spec.mod_id}-{uuid.uuid4().hex[:10]}"

        self._authorize(ToolAction.SCAFFOLD, staging, output_root, approved)
        try:
            generated = self.generator.generate(approved.spec, staging)
            self._write_json(staging / ".minecraft_ai" / "proposal.approved.json", approved.to_dict())
            self._write_json(
                staging / ".minecraft_ai" / "project-ir.json",
                self._project_ir(approved),
            )
            self._audit(
                staging,
                "fabric.scaffold",
                approved,
                {"changed_paths": len(generated.files), "status": "succeeded"},
            )

            self._authorize(ToolAction.VALIDATE, staging, output_root, approved)
            validation = self.validator.validate(staging, approved.spec)
            self._write_json(
                staging / ".minecraft_ai" / "validation-report.json",
                validation.to_dict(),
            )
            self._audit(
                staging,
                "quality.validate",
                approved,
                {
                    "status": "succeeded" if validation.passed else "failed",
                    "checks_run": validation.checks_run,
                    "finding_count": len(validation.findings),
                },
            )
            if not validation.passed:
                raise RuntimeError("Generated project failed deterministic validation.")
            staging.rename(project_root)
        except Exception:
            if staging.exists():
                failed = workspaces / f"failed-{staging.name.removeprefix('.staging-')}"
                if failed.exists():
                    shutil.rmtree(failed)
                staging.rename(failed)
            raise

        build_report = BuildReport(
            status="NOT_RUN",
            gradle_version=approved.spec.platform.gradle,
            commands=(),
            jar_path=None,
            gametest_report=None,
            error=None,
        )
        jar_report = ValidationReport(status="NOT_RUN", checks_run=0, findings=())
        validated_jar_sha256: str | None = None
        if build:
            self._authorize(ToolAction.GRADLE_BUILD, project_root, output_root, approved)
            if run_gametest:
                self._authorize(ToolAction.GAME_TEST, project_root, output_root, approved)
            runner = GradleRunner(resolved_gradle_cache)
            try:
                build_report = runner.build(project_root, run_gametest=run_gametest)
            except Exception as exc:
                build_report = BuildReport(
                    status="FAIL",
                    gradle_version=approved.spec.platform.gradle,
                    commands=(),
                    jar_path=None,
                    gametest_report=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            if build_report.jar_path:
                built_jar = Path(build_report.jar_path)
                candidate_jar = (
                    project_root
                    / ".minecraft_ai"
                    / "candidate"
                    / f"{approved.spec.mod_id}-{approved.spec.version}.jar"
                )
                candidate_jar.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(built_jar, candidate_jar)
                build_report = BuildReport(
                    status=build_report.status,
                    gradle_version=build_report.gradle_version,
                    commands=build_report.commands,
                    jar_path=str(candidate_jar.resolve()),
                    gametest_report=build_report.gametest_report,
                    error=build_report.error,
                )
            self._write_json(
                project_root / ".minecraft_ai" / "build-report.json",
                build_report.to_dict(),
            )
            self._audit(
                project_root,
                "build.gradle",
                approved,
                {
                    "status": (
                        "succeeded"
                        if any(
                            command.name == "clean_build" and command.exit_code == 0
                            for command in build_report.commands
                        )
                        else "failed"
                    ),
                    "commands": [command.name for command in build_report.commands],
                    "error": build_report.error,
                },
            )
            if run_gametest:
                self._audit(
                    project_root,
                    "test.gametest",
                    approved,
                    {
                        "status": (
                            "succeeded"
                            if self._gametest_passed(build_report, approved.spec)
                            else "failed"
                        ),
                        "report": build_report.gametest_report,
                    },
                )
            if build_report.jar_path:
                candidate_path = Path(build_report.jar_path)
                digest_before_validation = _sha256(candidate_path)
                jar_report = validate_jar(candidate_path, approved.spec)
                digest_after_validation = _sha256(candidate_path)
                if digest_before_validation != digest_after_validation:
                    jar_report = ValidationReport(
                        status="FAIL",
                        checks_run=jar_report.checks_run + 1,
                        findings=(
                            *jar_report.findings,
                            Finding(
                                "JAR_CHANGED_DURING_VALIDATION",
                                "error",
                                str(candidate_path),
                                "Candidate JAR bytes changed while validation was running.",
                            ),
                        ),
                    )
                else:
                    validated_jar_sha256 = digest_after_validation
                self._write_json(
                    project_root / ".minecraft_ai" / "candidate-jar.json",
                    {
                        "path": build_report.jar_path,
                        "sha256": validated_jar_sha256,
                        "validation_status": jar_report.status,
                    },
                )
            self._write_json(
                project_root / ".minecraft_ai" / "jar-validation-report.json",
                jar_report.to_dict(),
            )

        self._authorize(ToolAction.PACKAGE, project_root, output_root, approved)
        release_dir, release_zip, released_jar = self._package_release(
            approved,
            project_root,
            releases,
            validation,
            build_report,
            jar_report,
            validated_jar_sha256=validated_jar_sha256,
            existing_report=existing_report,
        )
        release_ready = (
            validation.passed
            and build_report.passed
            and jar_report.passed
            and self._gametest_passed(build_report, approved.spec)
        )
        return PipelineResult(
            status="VERIFIED" if release_ready else ("SOURCE_READY" if not build else "FAILED"),
            project_root=str(project_root),
            release_dir=str(release_dir),
            release_zip=str(release_zip),
            jar_path=str(released_jar) if released_jar else None,
            proposal_hash=approved.calculate_hash(),
            validation_status=validation.status,
            build_status=build_report.status,
            gametest_status=(
                "NOT_RUN"
                if not run_gametest or not build
                else (
                    "PASS"
                    if self._gametest_passed(build_report, approved.spec)
                    else "FAIL"
                )
            ),
            release_ready=release_ready,
            existing_input_kind=(
                existing_report.input_kind if existing_report is not None else None
            ),
            imported_source_snapshot_hash=(
                approved.imported_source_snapshot_hash or None
            ),
        )

    @staticmethod
    def _bind_existing_input(
        proposal: Proposal,
        report: ExistingProjectReport,
    ) -> Proposal:
        if proposal.imported_source_snapshot_hash:
            raise SpecValidationError(
                "Planner returned an unexpected imported source binding."
            )
        if report.loader is not None and report.loader != proposal.spec.platform.loader:
            raise SpecValidationError(
                "Existing project loader is incompatible with the pinned Fabric target."
            )
        if report.minecraft_versions and not any(
            proposal.spec.platform.minecraft_version in constraint
            for constraint in report.minecraft_versions
        ):
            raise SpecValidationError(
                "Existing project Minecraft constraint is incompatible with the "
                f"pinned {proposal.spec.platform.minecraft_version} target."
            )
        input_summary = (
            f"기존 입력 {report.archive_name!r}의 {report.input_kind} inventory를 "
            f"{report.source_snapshot_hash} 기준선으로 사용합니다."
        )
        limitations = (
            "업로드 ZIP은 실행하거나 덮어쓰지 않으며 별도 revision candidate만 생성합니다.",
            "현재 수직 슬라이스는 임의 기존 소스에 최소 unified diff를 적용하지 않습니다.",
        )
        exclusions = proposal.exclusions
        if report.jar_only:
            exclusions = (
                *exclusions,
                "JAR-only 입력의 소스 수정: metadata/inventory 분석만 가능합니다.",
            )
        deferred = (
            *proposal.deferred_requests,
            DeferredRequest(
                capability="minimal_existing_project_diff",
                reason=(
                    "기존 프로젝트 보존형 최소 patch에는 source graph, invariant "
                    "diff와 이전 전체 회귀 suite가 추가로 필요합니다."
                ),
                suggested_phase="revision-engine",
            ),
        )
        bound = replace(
            proposal,
            assumptions=(*proposal.assumptions, input_summary, *limitations),
            exclusions=exclusions,
            deferred_requests=deferred,
            acceptance_tests=(
                *proposal.acceptance_tests,
                "실행 시 기존 입력을 다시 검사하고 승인된 snapshot hash와 같아야 한다.",
            ),
            imported_source_snapshot_hash=report.source_snapshot_hash,
            approval_hash="",
        ).with_hash()
        bound.validate()
        return bound

    @staticmethod
    def _verify_existing_input_binding(
        proposal: Proposal,
        *,
        existing_input: str | Path | None,
    ) -> ExistingProjectReport | None:
        expected = proposal.imported_source_snapshot_hash
        if not expected:
            if existing_input is not None:
                raise SpecValidationError(
                    "An existing input was supplied, but the approved proposal did not bind it."
                )
            return None
        if existing_input is None:
            raise SpecValidationError(
                "The approved proposal binds an existing project, so the same ZIP is required."
            )
        report = inspect_existing_project_archive(existing_input)
        if report.source_snapshot_hash != expected:
            raise SpecValidationError(
                "Existing project snapshot changed after approval; inspect and approve again."
            )
        return report

    def _authorize(
        self,
        action: ToolAction,
        project_root: Path,
        output_root: Path,
        proposal: Proposal,
    ) -> None:
        request = approved_request(
            action,
            project_root=project_root,
            workspace_root=output_root,
            proposal=proposal,
        )
        self.broker.authorize(request, proposal)

    @staticmethod
    def _project_ir(proposal: Proposal) -> dict[str, object]:
        return project_ir(proposal)

    def _package_release(
        self,
        proposal: Proposal,
        project_root: Path,
        releases_root: Path,
        validation: ValidationReport,
        build_report: BuildReport,
        jar_report: ValidationReport,
        *,
        validated_jar_sha256: str | None,
        existing_report: ExistingProjectReport | None,
    ) -> tuple[Path, Path, Path | None]:
        spec = proposal.spec
        release_name = f"{spec.mod_id}-{spec.version}"
        final_release_dir = releases_root / release_name
        if final_release_dir.exists():
            raise FileExistsError(f"Release already exists: {final_release_dir}")
        release_dir = releases_root / f".staging-{release_name}-{uuid.uuid4().hex[:10]}"
        for subdir in (
            "binaries",
            "source",
            "world",
            "packs",
            "config",
            "docs",
            "evidence",
            "supply_chain",
            "art_sources",
        ):
            (release_dir / subdir).mkdir(parents=True, exist_ok=True)

        source_zip = release_dir / "source" / f"{release_name}-source.zip"
        self._zip_tree(project_root, source_zip, exclude_build=True)
        released_jar: Path | None = None
        gates_pass = (
            validation.passed
            and build_report.passed
            and jar_report.passed
            and validated_jar_sha256 is not None
            and self._gametest_passed(build_report, spec)
        )
        if gates_pass and build_report.jar_path:
            released_jar = release_dir / "binaries" / f"{release_name}.jar"
            candidate_jar = Path(build_report.jar_path)
            candidate_sha256 = _sha256(candidate_jar)
            if candidate_sha256 != validated_jar_sha256:
                raise RuntimeError("Candidate JAR changed after validation.")
            shutil.copy2(candidate_jar, released_jar)
            if _sha256(released_jar) != validated_jar_sha256:
                released_jar.unlink(missing_ok=True)
                raise RuntimeError("Released JAR bytes changed after validation.")

        if spec.boss is not None:
            art_root = project_root / ".minecraft_ai" / "art_sources"
            for path in sorted(art_root.glob(f"{spec.boss.entity_id}.*")):
                shutil.copy2(path, release_dir / "art_sources" / path.name)
            boss_texture = (
                project_root
                / f"src/main/resources/assets/{spec.mod_id}/textures/entity/"
                f"{spec.boss.entity_id}.png"
            )
            shutil.copy2(
                boss_texture,
                release_dir / "art_sources" / f"{spec.boss.entity_id}.png",
            )
            released_mtl = release_dir / "art_sources" / f"{spec.boss.entity_id}.mtl"
            released_mtl.write_text(
                "\n".join(
                    (
                        f"map_Kd {spec.boss.entity_id}.png"
                        if line.startswith("map_Kd ")
                        else line
                    )
                    for line in released_mtl.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
        if spec.arena is not None:
            world_root = project_root / ".minecraft_ai" / "world"
            for path in sorted(world_root.glob(f"{spec.arena.arena_id}*")):
                shutil.copy2(path, release_dir / "world" / path.name)
            self._make_arena_datapack(project_root, release_dir / "packs", spec)

        self._write_json(release_dir / "evidence" / "proposal.json", proposal.to_dict())
        self._write_json(
            release_dir / "evidence" / "evidence-snapshot.json",
            {
                "schema_version": "minecraft-mod-ai/evidence-snapshot-v1",
                "snapshot_hash": proposal.evidence_snapshot_hash,
                "retrieved_context_policy": "data_only",
                "sources": proposal.to_dict()["evidence_sources"],
            },
        )
        self._write_json(
            release_dir / "evidence" / "capability-manifest.json",
            capability_manifest(),
        )
        if existing_report is not None:
            self._write_json(
                release_dir / "evidence" / "imported-project-inventory.json",
                existing_report.to_dict(),
            )
        self._write_json(
            release_dir / "evidence" / "deterministic-validation.json",
            validation.to_dict(),
        )
        self._write_json(
            release_dir / "evidence" / "build-report.json",
            build_report.to_dict(),
        )
        self._write_json(
            release_dir / "evidence" / "jar-validation.json",
            jar_report.to_dict(),
        )
        for filename in ("project-ir.json", "receipts.jsonl", "audit.jsonl"):
            source = project_root / ".minecraft_ai" / filename
            if source.is_file():
                shutil.copy2(source, release_dir / "evidence" / filename)
        self._copy_logs(project_root, release_dir / "evidence")
        self._write_json(
            release_dir / "supply_chain" / "sbom.cdx.json",
            self._sbom(spec),
        )
        self._write_json(
            release_dir / "supply_chain" / "provenance.json",
            {
                "schema_version": "minecraft-mod-ai/provenance-v1",
                "generator": "minecraft-mod-ai/0.1.0",
                "proposal_hash": proposal.calculate_hash(),
                "platform_lock": asdict(spec.platform),
                "evidence_snapshot_hash": proposal.evidence_snapshot_hash,
                "capability_manifest_hash": proposal.capability_manifest_hash,
                "imported_source_snapshot_hash": (
                    proposal.imported_source_snapshot_hash or None
                ),
                "imported_archive_sha256": (
                    existing_report.archive_sha256
                    if existing_report is not None
                    else None
                ),
                "build_verified": build_report.passed,
                "gametest_verified": self._gametest_passed(build_report, spec),
                "jar_verified": jar_report.passed,
                "binary_sha256": (
                    _sha256(released_jar)
                    if gates_pass and released_jar is not None
                    else None
                ),
                "external_art_assets": [],
                "generated_asset_license": "MIT",
            },
        )
        (release_dir / "config" / "README.md").write_text(
            "# Config\n\n이 archetype은 별도 설정 파일을 요구하지 않습니다.\n",
            encoding="utf-8",
        )
        self._write_docs(release_dir / "docs", proposal, gates_pass)

        package_result = {
            "status": "succeeded",
            "release_name": release_name,
            "source_sha256": _sha256(source_zip),
            "binary_sha256": (
                _sha256(released_jar) if released_jar is not None else None
            ),
            "release_ready": gates_pass,
        }
        package_receipt = make_worker_receipt(
            node_id="release.package",
            worker="release-packager",
            proposal=proposal,
            result=package_result,
            evidence=(
                f"source_sha256:{package_result['source_sha256']}",
                f"release_ready:{str(gates_pass).lower()}",
            ),
            status="succeeded",
        )
        self._write_json(
            release_dir / "evidence" / "release-package-receipt.json",
            package_receipt.to_dict(),
        )

        manifest_entries = []
        for path in sorted(release_dir.rglob("*")):
            if path.is_file() and path.name != "artifact-manifest.json":
                manifest_entries.append(
                    {
                        "path": str(path.relative_to(release_dir)).replace("\\", "/"),
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
        self._write_json(
            release_dir / "evidence" / "artifact-manifest.json",
            {
                "schema_version": "minecraft-mod-ai/artifact-manifest-v1",
                "release_ready": gates_pass,
                "verification_status": "PASS" if gates_pass else "NOT_RELEASE_READY",
                "artifacts": manifest_entries,
            },
        )
        release_dir.rename(final_release_dir)
        if released_jar is not None:
            released_jar = final_release_dir / released_jar.relative_to(release_dir)
        release_zip = releases_root / f"{release_name}.zip"
        staging_zip = releases_root / f".staging-{release_name}-{uuid.uuid4().hex[:10]}.zip"
        self._zip_tree(final_release_dir, staging_zip, exclude_build=False)
        staging_zip.replace(release_zip)
        return final_release_dir, release_zip, released_jar

    @staticmethod
    def _make_arena_datapack(project_root: Path, packs_root: Path, spec: Any) -> None:
        if spec.arena is None:
            return
        destination = packs_root / f"{spec.mod_id}-{spec.arena.arena_id}-datapack.zip"
        pack_meta = {
            "pack": {
                "pack_format": 15,
                "description": f"{spec.mod_name} arena function (requires the mod)",
            }
        }
        source_function = (
            project_root
            / f"src/main/resources/data/{spec.mod_id}/functions/build_{spec.arena.arena_id}.mcfunction"
        )
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("pack.mcmeta", json.dumps(pack_meta, ensure_ascii=False, indent=2))
            archive.write(
                source_function,
                f"data/{spec.mod_id}/functions/build_{spec.arena.arena_id}.mcfunction",
            )

    @staticmethod
    def _sbom(spec: Any) -> dict[str, object]:
        components = [
            ("library", "Fabric Loader", f"pkg:maven/net.fabricmc/fabric-loader@{spec.platform.fabric_loader}"),
            ("library", "Fabric API", f"pkg:maven/net.fabricmc.fabric-api/fabric-api@{spec.platform.fabric_api}"),
            ("framework", "Fabric Loom", f"pkg:maven/net.fabricmc/fabric-loom@{spec.platform.fabric_loom}"),
            ("application", "Minecraft", f"pkg:generic/minecraft@{spec.platform.minecraft_version}"),
        ]
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {"component": {"type": "application", "name": spec.mod_id, "version": spec.version}},
            "components": [
                {"type": kind, "name": name, "purl": purl}
                for kind, name, purl in components
            ],
        }

    @staticmethod
    def _write_docs(docs_root: Path, proposal: Proposal, gates_pass: bool) -> None:
        spec = proposal.spec
        boss_text = (
            f"\n- 보스: `{spec.boss.entity_id}` ({spec.boss.display_name_ko})"
            if spec.boss
            else ""
        )
        arena_text = (
            f"\n- 아레나 생성: `/function {spec.mod_id}:build_{spec.arena.arena_id}` "
            "(권한 있는 테스트 서버에서 명시적으로 실행)"
            if spec.arena
            else ""
        )
        (docs_root / "MOD_DESCRIPTION_KO.md").write_text(
            f"""# {spec.mod_name}

{spec.summary}

- 대상: Minecraft Java {spec.platform.minecraft_version} / Fabric / Java 17
- 릴리스 게이트: {"PASS" if gates_pass else "NOT RELEASE READY"}
{boss_text}{arena_text}

`NOT RELEASE READY`인 번들은 소스와 실패 증거만 제공하며 설치용 JAR를 포함하지 않습니다.
""",
            encoding="utf-8",
        )
        (docs_root / "INSTALL_KO.md").write_text(
            f"""# 설치

1. Minecraft {spec.platform.minecraft_version}용 Fabric Loader를 설치합니다.
2. 호환 Fabric API와 검증된 `{spec.mod_id}-{spec.version}.jar`를 `mods`에 넣습니다.
3. 실패/소스 전용 번들의 JAR는 설치하지 마세요.
""",
            encoding="utf-8",
        )
        (docs_root / "ADMIN_KO.md").write_text(
            f"""# 관리자 가이드

아레나 함수는 자동 실행되지 않습니다.
{arena_text or "- 이 릴리스에는 아레나가 없습니다."}

실제 월드에서 실행하기 전 백업과 테스트 월드 검증을 권장합니다.
""",
            encoding="utf-8",
        )

    @staticmethod
    def _copy_logs(project_root: Path, evidence_root: Path) -> None:
        log_root = project_root / ".minecraft_ai" / "logs"
        if not log_root.is_dir():
            return
        for path in sorted(log_root.glob("*.log")):
            shutil.copy2(path, evidence_root / path.name)

    @staticmethod
    def _zip_tree(root: Path, destination: Path, *, exclude_build: bool) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if exclude_build and (
                    relative.parts[0] in {"build", ".gradle", "run"}
                    or "gradle-user-home" in relative.parts
                    or relative.parts[:2] == (".minecraft_ai", "candidate")
                ):
                    continue
                archive.write(path, str(relative).replace("\\", "/"))

    @staticmethod
    def _gametest_passed(build_report: BuildReport, spec: Any) -> bool:
        command_passed = any(
            command.name == "gametest" and command.exit_code == 0
            for command in build_report.commands
        )
        if not command_passed or not build_report.gametest_report:
            return False
        report_path = Path(build_report.gametest_report)
        if not report_path.is_file():
            return False
        try:
            root = ET.parse(report_path).getroot()
        except (ET.ParseError, OSError):
            return False
        testcases = list(root.iter("testcase"))
        if not testcases:
            return False
        for suite in root.iter("testsuite"):
            for aggregate in ("failures", "errors", "skipped"):
                value = suite.attrib.get(aggregate)
                if value is not None:
                    try:
                        if int(value) != 0:
                            return False
                    except ValueError:
                        return False
        if any(
            testcase.find("failure") is not None
            or testcase.find("error") is not None
            or testcase.find("skipped") is not None
            for testcase in testcases
        ):
            return False
        main_class = "".join(part.capitalize() for part in spec.mod_id.split("_")) + "Mod"
        expected_name = f"{main_class}GameTests.generatedRegistriesAreLive".lower()
        return any(
            testcase.attrib.get("name", "").lower() == expected_name
            for testcase in testcases
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _audit(project_root: Path, action: str, proposal: Proposal, result: dict[str, object]) -> None:
        path = project_root / ".minecraft_ai" / "audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        node_by_action = {
            "fabric.scaffold": "fabric.scaffold",
            "quality.validate": "quality.source.validate",
            "build.gradle": "build.gradle",
            "test.gametest": "test.gametest",
            "release.package": "release.package",
        }
        worker_by_action = {
            "fabric.scaffold": "fabric-generator",
            "quality.validate": "independent-source-validator",
            "build.gradle": "gradle-runner",
            "test.gametest": "gametest-runner",
            "release.package": "release-packager",
        }
        raw_status = str(result.get("status", "failed"))
        status = "succeeded" if raw_status == "succeeded" else "failed"
        evidence = tuple(
            f"{key}:{value}"
            for key, value in sorted(result.items())
            if key not in {"status", "error"}
            and value is not None
            and value != ""
            and value != ()
            and value != []
        )
        if not evidence:
            evidence = (f"action:{action}",)
        error = None
        if status == "failed":
            error = str(result.get("error") or f"{action} gate failed")
        receipt = make_worker_receipt(
            node_id=node_by_action[action],
            worker=worker_by_action[action],
            proposal=proposal,
            result=result,
            evidence=evidence,
            status=status,
            error=error,
        )
        event = {
            "schema_version": "minecraft-mod-ai/audit-event-v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "proposal_hash": proposal.calculate_hash(),
            "receipt_id": receipt.receipt_id,
            "result": result,
        }
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        receipt_path = project_root / ".minecraft_ai" / "receipts.jsonl"
        with receipt_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(receipt_json_line(receipt) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
