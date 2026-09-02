from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
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
from .publisher import (
    dependency_inventory_from_metadata,
    fabric_dependency_components,
    read_fabric_metadata_file,
)
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
        """Create a target-bound in-memory proposal with zero project writes."""
        proposal = self.planner.plan(prompt)
        report: ExistingProjectReport | None = None
        if existing_input is not None:
            report = inspect_existing_project_archive(existing_input)

        from .platform_resolver import resolve_platform, retarget_proposal

        selection = resolve_platform(
            prompt,
            existing_version=(report.minecraft_version if report is not None else None),
            existing_loader=(report.loader if report is not None else None),
        )
        proposal = retarget_proposal(proposal, selection)
        if report is not None:
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
        fabric_metadata_path = project_root / "src/main/resources/fabric.mod.json"
        fabric_metadata = read_fabric_metadata_file(fabric_metadata_path)
        fabric_dependencies = dependency_inventory_from_metadata(
            fabric_metadata,
            platform_lock=spec.platform,
        )
        release_dir = releases_root / f".staging-{release_name}-{uuid.uuid4().hex[:10]}"
        for subdir in (
            "binaries",
            "source",
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

        art_source_root = project_root / ".minecraft_ai" / "art_sources"
        if art_source_root.is_dir() and not art_source_root.is_symlink():
            for source in sorted(art_source_root.iterdir()):
                if not source.is_file() or source.is_symlink():
                    continue
                destination = release_dir / "art_sources" / source.name
                if source.suffix.casefold() == ".mtl" and spec.boss is not None:
                    texture_name = f"{spec.boss.entity_id}.png"
                    original = source.read_text(encoding="utf-8")
                    rewritten = "\n".join(
                        (
                            f"map_Kd {texture_name}"
                            if line.lstrip().startswith("map_Kd ")
                            else line
                        )
                        for line in original.splitlines()
                    )
                    if original.endswith("\n"):
                        rewritten += "\n"
                    self._write_text(destination, rewritten)
                else:
                    shutil.copy2(source, destination)
        if spec.boss is not None:
            boss_texture = (
                project_root
                / "src/main/resources/assets"
                / spec.mod_id
                / "textures/entity"
                / f"{spec.boss.entity_id}.png"
            )
            if boss_texture.is_file() and not boss_texture.is_symlink():
                shutil.copy2(
                    boss_texture,
                    release_dir / "art_sources" / boss_texture.name,
                )

        released_jar: Path | None = None
        gates_pass = (
            validation.passed
            and build_report.passed
            and jar_report.passed
            and self._gametest_passed(build_report, spec)
        )
        if gates_pass:
            if build_report.jar_path is None or validated_jar_sha256 is None:
                raise RuntimeError("Verified release is missing a validated candidate JAR.")
            candidate_jar = Path(build_report.jar_path)
            if _sha256(candidate_jar) != validated_jar_sha256:
                raise RuntimeError(
                    "Candidate JAR changed after validation; refusing to promote binary."
                )
            released_jar = release_dir / "binaries" / candidate_jar.name
            shutil.copy2(candidate_jar, released_jar)
            if _sha256(released_jar) != validated_jar_sha256:
                raise RuntimeError(
                    "Promoted JAR digest does not match the validated candidate."
                )
        self._write_json(
            release_dir / "evidence" / "proposal.approved.json", proposal.to_dict()
        )
        self._write_json(
            release_dir / "evidence" / "validation-report.json", validation.to_dict()
        )
        self._write_json(
            release_dir / "evidence" / "jar-validation-report.json", jar_report.to_dict()
        )
        self._write_json(
            release_dir / "evidence" / "build-report.json", build_report.to_dict()
        )
        self._write_json(
            release_dir / "evidence" / "runtime-gate.json",
            {
                "release_ready": gates_pass,
                "gametest_passed": self._gametest_passed(build_report, spec),
                "jar_validation_passed": jar_report.passed,
                "static_validation_passed": validation.passed,
                "validated_jar_sha256": validated_jar_sha256,
            },
        )
        self._write_json(
            release_dir / "supply_chain" / "fabric-dependencies.json",
            {
                "components": fabric_dependency_components(fabric_dependencies),
                "dependencies": list(fabric_dependencies),
            },
        )
        self._write_text(
            release_dir / "supply_chain" / "sbom.cdx.json",
            self._cyclonedx_sbom(proposal, fabric_metadata),
        )
        self._write_json(
            release_dir / "supply_chain" / "provenance.json",
            {
                "schema_version": "mmm/distribution-provenance-v1",
                "source_sha256": "sha256:" + _sha256(source_zip),
                "binary_sha256": (
                    "sha256:" + _sha256(released_jar)
                    if released_jar is not None
                    else None
                ),
                "platform_lock": spec.platform.to_dict(),
                "environment": fabric_metadata.get("environment"),
                "declared_fabric_dependencies": list(fabric_dependencies),
            },
        )
        self._write_text(
            release_dir / "supply_chain" / "provenance.intoto.jsonl",
            self._provenance_statement(proposal, source_zip, released_jar),
        )
        self._write_text(
            release_dir / "docs" / "README.md",
            self._release_readme(proposal, validation, build_report, jar_report),
        )
        self._write_text(
            release_dir / "docs" / "CAPABILITIES.json",
            json.dumps(
                capability_manifest(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        if existing_report is not None:
            inventory = existing_report.to_dict()
            self._write_json(
                release_dir / "evidence" / "existing-input-report.json",
                inventory,
            )
            self._write_json(
                release_dir / "evidence" / "imported-project-inventory.json",
                inventory,
            )
        self._zip_tree(release_dir, release_zip := releases_root / f"{release_name}.zip")
        release_dir.rename(final_release_dir)
        return final_release_dir, release_zip, released_jar

    @staticmethod
    def _gametest_passed(build_report: BuildReport, spec) -> bool:
        del spec
        gametest_commands = tuple(
            command for command in build_report.commands if command.name == "gametest"
        )
        if not gametest_commands:
            return True
        if not all(
            command.exit_code == 0 and not command.timed_out
            for command in gametest_commands
        ):
            return False
        if not build_report.gametest_report:
            return False
        report_path = Path(build_report.gametest_report)
        if not report_path.is_file() or report_path.is_symlink():
            return False
        try:
            from xml.etree import ElementTree

            root = ElementTree.parse(report_path).getroot()
        except (OSError, ElementTree.ParseError):
            return False

        testcases = 0
        for element in root.iter():
            tag = str(element.tag).rsplit("}", 1)[-1]
            if tag == "testcase":
                testcases += 1
            elif tag in {"failure", "error", "skipped"}:
                return False
        for element in root.iter():
            tag = str(element.tag).rsplit("}", 1)[-1]
            if tag not in {"testsuite", "testsuites"}:
                continue
            for field in ("failures", "errors", "skipped"):
                raw = str(element.attrib.get(field, "0") or "0").strip()
                try:
                    if int(raw) != 0:
                        return False
                except ValueError:
                    return False
        return testcases > 0

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        MinecraftModPipeline._write_text(
            path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def _zip_tree(root: Path, destination: Path, *, exclude_build: bool = False) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                if exclude_build and (
                    relative.parts[0] == "build"
                    or relative.parts[0] == ".gradle"
                    or (relative.parts[0] == ".minecraft_ai" and "candidate" in relative.parts)
                ):
                    continue
                archive.write(path, relative.as_posix())

    @staticmethod
    def _audit(
        project_root: Path,
        worker: str,
        proposal: Proposal,
        details: dict[str, object],
    ) -> None:
        path = project_root / ".minecraft_ai" / "audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = make_worker_receipt(
            worker=worker,
            worker_kind="local_tool",
            task_id=worker,
            status=str(details.get("status", "succeeded")),
            scope=worker,
            inputs=(proposal.calculate_hash(),),
            outputs=tuple(
                str(item) for item in details.get("commands", ())
            )
            if isinstance(details.get("commands"), list)
            else (),
            tools=(worker,),
            validations=(str(details.get("status", "succeeded")),),
            error=str(details.get("error") or "") or None,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(receipt_json_line(receipt))
            handle.write("\n")

    @staticmethod
    def _cyclonedx_sbom(proposal: Proposal, fabric_metadata: dict[str, Any]) -> str:
        components = [
            {
                "type": "application",
                "name": proposal.spec.mod_name,
                "version": proposal.spec.version,
                "bom-ref": f"pkg:generic/{proposal.spec.mod_id}@{proposal.spec.version}",
            }
        ]
        dependencies = dependency_inventory_from_metadata(
            fabric_metadata,
            platform_lock=proposal.spec.platform,
        )
        components.extend(fabric_dependency_components(dependencies))
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": components[0],
            },
            "components": components,
        }
        return json.dumps(bom, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @staticmethod
    def _provenance_statement(
        proposal: Proposal,
        source_zip: Path,
        jar_path: Path | None,
    ) -> str:
        subjects = [{"name": source_zip.name, "digest": {"sha256": _sha256(source_zip)}}]
        if jar_path is not None:
            subjects.append({"name": jar_path.name, "digest": {"sha256": _sha256(jar_path)}})
        statement = {
            "_type": "https://in-toto.io/Statement/v1",
            "subject": subjects,
            "predicateType": "https://slsa.dev/provenance/v1",
            "predicate": {
                "buildDefinition": {
                    "buildType": "https://example.invalid/minecraft-mod-ai/local-pipeline/v1",
                    "externalParameters": {
                        "proposal_hash": proposal.calculate_hash(),
                        "platform": proposal.spec.platform.to_dict(),
                    },
                    "resolvedDependencies": [],
                },
                "runDetails": {
                    "builder": {"id": "minecraft-mod-ai/local-policy-broker"},
                    "metadata": {"invocationId": str(uuid.uuid4())},
                },
            },
        }
        return json.dumps(statement, ensure_ascii=False, sort_keys=True) + "\n"

    @staticmethod
    def _release_readme(
        proposal: Proposal,
        validation: ValidationReport,
        build_report: BuildReport,
        jar_report: ValidationReport,
    ) -> str:
        lines = [
            f"# {proposal.spec.mod_name}",
            "",
            f"Proposal: `{proposal.calculate_hash()}`",
            f"Platform: Minecraft {proposal.spec.platform.minecraft_version} / "
            f"{proposal.spec.platform.loader} / Java {proposal.spec.platform.java_version}",
            f"Static validation: {validation.status}",
            f"Build: {build_report.status}",
            f"JAR validation: {jar_report.status}",
            f"GameTest: {build_report.gametest_report or 'NOT_RUN'}",
            "",
            "## Capability limits",
            "",
            "See `CAPABILITIES.json` and the approved proposal exclusions/deferred requests.",
        ]
        return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["MinecraftModPipeline", "PipelineResult"]
