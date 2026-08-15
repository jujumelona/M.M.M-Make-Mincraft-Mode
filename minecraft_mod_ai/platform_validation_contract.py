from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

from .imported_platform_repair import read_valid_marker
from .platform_catalog import adapter_for_lock_values, adapter_from_project


def install(module: Any) -> None:
    validator = module.ProjectValidator
    if getattr(validator.validate, "_mmm_dynamic_platform_validation", False):
        return

    original_validate = validator.validate

    @wraps(original_validate)
    def validate(self: Any, root: Path, spec: Any):
        root = root.expanduser().resolve()
        try:
            expected = adapter_for_lock_values(spec.platform)
        except Exception as exc:
            return _platform_failure(
                module,
                "PLATFORM_LOCK_INVALID",
                f"Approved platform lock is invalid: {exc}",
            )

        try:
            actual = adapter_from_project(root)
        except Exception as exc:
            marker = _authorized_import_repair_marker(
                module,
                root,
                spec,
                expected,
            )
            if marker is None:
                return _platform_failure(
                    module,
                    "PLATFORM_LOCK_INVALID",
                    f"Project target is missing, mixed or unsupported: {exc}",
                )
            # An approved source import may enter repair with incomplete exact
            # Gradle/Yarn/Fabric metadata, but the marker is not compatibility or
            # release evidence. Preserve every ordinary source/security check.
            if expected.source_api_family == "fabric_live_ai":
                report = _validate_live_project(
                    self,
                    module,
                    root,
                    spec,
                    expected,
                    allow_toolchain_repair=True,
                )
            else:
                report = _validate_reviewed_project(
                    self,
                    module,
                    root,
                    spec,
                    expected,
                    original_validate,
                )
            findings = list(report.findings)
            findings.append(
                module.Finding(
                    "PLATFORM_REPAIR_REQUIRED",
                    "warning",
                    ".minecraft_ai/imported-platform-repair.json",
                    (
                        "Imported source matches the approved Minecraft/loader target, "
                        "but its exact toolchain must be repaired and re-resolved before release."
                    ),
                )
            )
            return module.ValidationReport(
                status=(
                    "PASS"
                    if not any(item.severity == "error" for item in findings)
                    else "FAIL"
                ),
                checks_run=report.checks_run + 1,
                findings=tuple(findings),
            )

        if actual.adapter_id != expected.adapter_id:
            return _platform_failure(
                module,
                "PLATFORM_LOCK_MISMATCH",
                (
                    f"Project uses {actual.adapter_id} but approved proposal uses "
                    f"{expected.adapter_id}."
                ),
            )

        if expected.source_api_family == "fabric_live_ai":
            return _validate_live_project(self, module, root, spec, expected)
        return _validate_reviewed_project(
            self,
            module,
            root,
            spec,
            expected,
            original_validate,
        )

    validate._mmm_dynamic_platform_validation = True
    validator.validate = validate


def _platform_failure(module: Any, code: str, message: str):
    return module.ValidationReport(
        status="FAIL",
        checks_run=1,
        findings=(
            module.Finding(
                code,
                "error",
                ".minecraft_ai/platform-lock.json",
                message,
            ),
        ),
    )


def _authorized_import_repair_marker(
    module: Any,
    root: Path,
    spec: Any,
    adapter: Any,
) -> dict[str, Any] | None:
    findings: list[Any] = []
    try:
        complete = module._load_complete_project_proposal(root, spec, findings)
    except Exception:
        return None
    if complete is None or findings:
        return None
    archive_sha256 = str(getattr(complete, "existing_input_sha256", ""))
    if not archive_sha256:
        return None
    return read_valid_marker(
        root,
        adapter=adapter,
        archive_sha256=archive_sha256,
    )


def _validate_reviewed_project(
    self: Any,
    module: Any,
    root: Path,
    spec: Any,
    expected: Any,
    original_validate: Any,
):
    report = original_validate(self, root, spec)
    extra = list(report.findings)
    checks = report.checks_run
    pack = root / "src/main/resources/pack.mcmeta"
    checks += 1
    payload = self._load_json(pack, extra, root)
    pack_data = payload.get("pack") if isinstance(payload, dict) else None
    if (
        not isinstance(pack_data, dict)
        or pack_data.get("pack_format") != expected.resource_pack_format
    ):
        extra.append(
            module.Finding(
                "BAD_RESOURCE_PACK_FORMAT",
                "error",
                self._rel(root, pack),
                (
                    f"pack_format must equal {expected.resource_pack_format} for "
                    f"Minecraft {expected.minecraft_version}."
                ),
            )
        )
    return module.ValidationReport(
        status=(
            "PASS" if not any(item.severity == "error" for item in extra) else "FAIL"
        ),
        checks_run=checks,
        findings=tuple(extra),
    )


def _reviewed_live_network_sources(
    module: Any,
    root: Path,
    spec: Any,
    findings: list[Any],
) -> dict[str, str]:
    """Reuse the base validator's exact sidecar-source exception when available."""

    try:
        complete = module._load_complete_project_proposal(root, spec, findings)
        return module._reviewed_local_ai_sidecar_network_sources(
            root,
            spec,
            complete,
        )
    except Exception:
        return {}


def _validate_live_project(
    self: Any,
    module: Any,
    root: Path,
    spec: Any,
    adapter: Any,
    *,
    allow_toolchain_repair: bool = False,
):
    """Version-neutral source gate for officially bootstrapped future targets."""

    findings: list[Any] = []
    checks = 0
    if not root.is_dir() or root.is_symlink():
        return module.ValidationReport(
            status="FAIL",
            checks_run=1,
            findings=(
                module.Finding(
                    "PROJECT_ROOT_INVALID",
                    "error",
                    str(root),
                    "Project root must be a regular directory.",
                ),
            ),
        )

    for path in sorted(root.rglob("*")):
        checks += 1
        relative = self._rel(root, path)
        if path.is_symlink():
            findings.append(
                module.Finding(
                    "SYMLINK",
                    "error",
                    relative,
                    "Symlinks are not allowed.",
                )
            )
            continue
        try:
            path.resolve().relative_to(root)
        except ValueError:
            findings.append(
                module.Finding(
                    "PATH_ESCAPE",
                    "error",
                    relative,
                    "Path escaped the staging root.",
                )
            )
            continue
        if path.is_file() and path.stat().st_size > self.policy.max_single_file_bytes:
            findings.append(
                module.Finding(
                    "FILE_TOO_LARGE",
                    "error",
                    relative,
                    "File exceeds MMM_MAX_SINGLE_FILE_BYTES host resource policy.",
                )
            )

    for path in sorted({*root.rglob("*.json"), *root.rglob("*.mcmeta")}):
        checks += 1
        self._load_json(path, findings, root)

    reviewed_network_sources = _reviewed_live_network_sources(
        module,
        root,
        spec,
        findings,
    )
    java_files = sorted(root.rglob("*.java"))
    checks += 1
    if not java_files:
        findings.append(
            module.Finding(
                "NO_JAVA",
                "error",
                ".",
                "No Java sources were generated.",
            )
        )
    for path in java_files:
        checks += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = self._rel(root, path)
        for token in self.FORBIDDEN_JAVA:
            if token in text:
                if (
                    token == "java.net."
                    and reviewed_network_sources.get(relative) == text
                ):
                    continue
                findings.append(
                    module.Finding(
                        "FORBIDDEN_JAVA_API",
                        "error",
                        relative,
                        f"Generated source contains forbidden API token {token!r}.",
                    )
                )

    if not allow_toolchain_repair:
        properties = _read_properties(root / "gradle.properties")
        expected_properties = {
            "minecraft_version": adapter.minecraft_version,
            "loader_version": adapter.fabric_loader,
            "fabric_api_version": adapter.fabric_api,
            "loom_version": adapter.fabric_loom,
        }
        for key, expected_value in expected_properties.items():
            checks += 1
            actual_value = properties.get(key)
            if key == "fabric_api_version" and actual_value is None:
                actual_value = properties.get("fabric_version")
            if actual_value != expected_value:
                findings.append(
                    module.Finding(
                        "LIVE_TOOLCHAIN_MISMATCH",
                        "error",
                        "gradle.properties",
                        f"{key} must match the approved official target {expected_value!r}.",
                    )
                )

    fabric_path = root / "src/main/resources/fabric.mod.json"
    fabric = self._load_json(fabric_path, findings, root)
    checks += 1
    if isinstance(fabric, dict):
        for key, expected_value in (
            ("id", spec.mod_id),
            ("version", "${version}"),
            ("environment", "*"),
        ):
            checks += 1
            if fabric.get(key) != expected_value:
                findings.append(
                    module.Finding(
                        "BAD_FABRIC_METADATA",
                        "error",
                        self._rel(root, fabric_path),
                        f"{key} must equal {expected_value!r}.",
                    )
                )
        entrypoints = fabric.get("entrypoints")
        checks += 1
        if not isinstance(entrypoints, dict) or not _nonempty_entrypoint(entrypoints.get("main")):
            findings.append(
                module.Finding(
                    "BAD_ENTRYPOINTS",
                    "error",
                    self._rel(root, fabric_path),
                    "Live Fabric project must expose at least one main entrypoint.",
                )
            )
        depends = fabric.get("depends")
        checks += 1
        if not isinstance(depends, dict):
            findings.append(
                module.Finding(
                    "BAD_FABRIC_DEPENDS",
                    "error",
                    self._rel(root, fabric_path),
                    "depends must be an object.",
                )
            )
        else:
            required_ids = {"fabricloader", "minecraft", "java"}
            if not ({"fabric-api", "fabric"} & set(depends)):
                required_ids.add("fabric-api")
            for dependency_id in sorted(required_ids):
                checks += 1
                if dependency_id == "fabric-api" and (
                    "fabric-api" in depends or "fabric" in depends
                ):
                    continue
                value = depends.get(dependency_id)
                if not isinstance(value, str) or not value.strip():
                    findings.append(
                        module.Finding(
                            "BAD_FABRIC_DEPENDS",
                            "error",
                            self._rel(root, fabric_path),
                            f"Missing non-empty dependency predicate for {dependency_id}.",
                        )
                    )

    status = "PASS" if not any(item.severity == "error" for item in findings) else "FAIL"
    return module.ValidationReport(
        status=status,
        checks_run=checks,
        findings=tuple(findings),
    )


def _read_properties(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _nonempty_entrypoint(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_nonempty_entrypoint(item) for item in value)
    if isinstance(value, dict):
        return any(_nonempty_entrypoint(item) for item in value.values())
    return False
