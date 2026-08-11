from __future__ import annotations

import json
import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "mmm/clean-room-build-v1"


class CleanRoomBuildError(ValueError):
    pass


def _copy_source(
    source: Path,
    destination: Path,
    validation_module: Any,
) -> None:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            if validation_module._is_build_input(relative):
                raise CleanRoomBuildError(
                    f"Clean-room verification refused project symlink: {relative}"
                )
            continue
        if path.is_dir() or not validation_module._is_build_input(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _receipt_path(run_root: Path, fingerprint: str) -> Path:
    return (
        run_root
        / ".minecraft_ai"
        / "clean-room"
        / "receipts"
        / f"{fingerprint}.json"
    )


def clean_room_build(
    *,
    approved: Any,
    run_root: Path,
    project_root: Path,
    build_report: Mapping[str, Any],
    orchestrator_module: Any,
    validation_module: Any,
) -> dict[str, Any]:
    """Build an exact source snapshot with no live build/.gradle outputs.

    Dependency and Gradle build caches remain reusable. The proof is therefore a
    clean-source reproducibility build, not a deliberately cold dependency download.
    """

    fingerprint = validation_module.project_build_fingerprint(project_root)
    receipt_path = _receipt_path(run_root, fingerprint)
    if receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            cached = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == SCHEMA
            and cached.get("source_fingerprint") == fingerprint
            and cached.get("status") == "PASS"
        ):
            jar = Path(str(cached.get("jar_path", "")))
            if jar.is_file() and not jar.is_symlink():
                return cached

    clean_root = (
        run_root
        / ".minecraft_ai"
        / "clean-room"
        / "workspaces"
        / fingerprint[:20]
    )
    _copy_source(project_root, clean_root, validation_module)
    report = orchestrator_module.GradleRunner(
        run_root / ".cache" / "gradle"
    ).build(
        clean_root,
        run_gametest=False,
    ).to_dict()

    status = "FAIL"
    jar_validation: dict[str, Any] | None = None
    jar_path = ""
    jar_sha256 = ""
    if report.get("status") == "PASS":
        raw = report.get("jar_path")
        if isinstance(raw, str):
            jar = Path(raw).expanduser().resolve()
            if jar.is_file() and not jar.is_symlink():
                jar_validation = orchestrator_module.validate_jar(
                    jar,
                    approved.base_proposal.spec,
                ).to_dict()
                if jar_validation.get("status") == "PASS":
                    status = "PASS"
                    jar_path = str(jar)
                    jar_sha256 = (
                        orchestrator_module.CompleteProductionOrchestrator._file_hash(
                            jar
                        )
                    )

    receipt = {
        "schema_version": SCHEMA,
        "status": status,
        "source_fingerprint": fingerprint,
        "live_build_status": str(build_report.get("status", "")),
        "build": report,
        "jar_validation": jar_validation,
        "jar_path": jar_path,
        "jar_sha256": jar_sha256,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def install(
    orchestrator_module: Any,
    quality_evidence_module: Any,
    validation_module: Any,
) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator
    original_evaluate = cls._evaluate_quality
    if not getattr(original_evaluate, "_mmm_clean_room_quality", False):

        @wraps(original_evaluate)
        def evaluate(self: Any, *args: Any, **kwargs: Any):
            build_report = kwargs.get("build_report")
            approved = kwargs.get("approved")
            run_root = kwargs.get("run_root")
            project_root = kwargs.get("project_root")
            if (
                isinstance(build_report, dict)
                and build_report.get("status") == "PASS"
                and approved is not None
                and isinstance(run_root, Path)
                and isinstance(project_root, Path)
            ):
                clean = clean_room_build(
                    approved=approved,
                    run_root=run_root,
                    project_root=project_root,
                    build_report=build_report,
                    orchestrator_module=orchestrator_module,
                    validation_module=validation_module,
                )
                augmented = dict(build_report)
                augmented["clean_room_build"] = clean
                kwargs["build_report"] = augmented
            return original_evaluate(self, *args, **kwargs)

        evaluate._mmm_clean_room_quality = True
        cls._evaluate_quality = evaluate

    def clean_build_evidence(value: Mapping[str, Any] | None):
        if not isinstance(value, Mapping) or value.get("status") != "PASS":
            return None
        clean = value.get("clean_room_build")
        if (
            not isinstance(clean, Mapping)
            or clean.get("schema_version") != SCHEMA
            or clean.get("status") != "PASS"
            or not isinstance(clean.get("source_fingerprint"), str)
            or not clean.get("source_fingerprint")
        ):
            return None
        nested = clean.get("build")
        jar_validation = clean.get("jar_validation")
        if (
            not isinstance(nested, Mapping)
            or nested.get("status") != "PASS"
            or not isinstance(jar_validation, Mapping)
            or jar_validation.get("status") != "PASS"
        ):
            return None
        commands = quality_evidence_module._commands(nested)
        builds = [
            command
            for command in commands
            if command.get("name") == "build"
        ]
        if (
            not builds
            or not all(
                quality_evidence_module._command_passed(command)
                for command in builds
            )
        ):
            return None
        raw_path = clean.get("jar_path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        digest = quality_evidence_module._regular_file_sha256(Path(raw_path))
        if digest is None or digest != clean.get("jar_sha256"):
            return None
        facts = {
            "schema_version": SCHEMA,
            "source_fingerprint": clean["source_fingerprint"],
            "jar_sha256": digest,
            "build_command_count": len(builds),
            "jar_checks": jar_validation.get("checks_run", 0),
        }
        return [
            quality_evidence_module._evidence_ref(
                "clean-room-build",
                facts,
            )
        ], [clean]

    clean_build_evidence._mmm_clean_room_required = True
    quality_evidence_module._clean_build_evidence = clean_build_evidence

    original_command = cls._command_receipt_passed
    if not getattr(original_command, "_mmm_incremental_gate_alias", False):
        def command_receipt_passed(
            build_report: dict[str, Any] | None,
            name: str,
        ) -> bool:
            # Inner repair/build loops stay incremental. The build quality
            # dimension above is stricter and requires the clean-room proof.
            actual = "build" if name == "clean_build" else name
            return original_command(build_report, actual)

        command_receipt_passed._mmm_incremental_gate_alias = True
        cls._command_receipt_passed = staticmethod(command_receipt_passed)
