from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_VALIDATION_CHECKPOINT_FAMILIES = {
    "validate-source": "validate-source",
    "validate-source-final": "validate-source",
    "validate-jdt": "validate-jdt",
    "validate-jdt-final": "validate-jdt",
    "validate-jar": "validate-jar",
}


def _canonical_validation_checkpoint(checkpoint_id: str) -> str:
    family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)
    if family is None:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")
    return family


def _file_digest(module: Any) -> str:
    path_value = getattr(module, "__file__", "")
    if not path_value:
        return "missing"
    try:
        return hashlib.sha256(Path(path_value).resolve().read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _validation_modules(checkpoint_id: str) -> tuple[Any, ...]:
    """Return every module whose bytes can change one cached validation decision."""

    from . import complete_orchestrator

    common: list[Any] = [
        sys.modules[__name__],
        complete_orchestrator,
    ]
    if checkpoint_id == "validate-source":
        from . import (
            platform_validation_contract,
            scalable_validator,
            scale_policy,
            validator,
        )

        common.extend(
            (
                scalable_validator,
                validator,
                scale_policy,
                platform_validation_contract,
            )
        )
    elif checkpoint_id == "validate-jar":
        from . import scale_policy, toolchain_contract, validator

        common.extend((validator, scale_policy, toolchain_contract))
    else:
        from . import (
            java_core,
            jvm_owner_bootstrap,
            project_model,
            validation_diagnostic_contract,
        )

        common.extend(
            (
                java_core,
                jvm_owner_bootstrap,
                project_model,
                validation_diagnostic_contract,
            )
        )
    return tuple(common)


def validation_implementation_fingerprint(checkpoint_id: str) -> str:
    """Hash the complete active validation implementation and MMM host policy.

    Validation checkpoints are reusable only when their generated inputs, validation
    implementation (including runtime-installed validation contracts), bootstrap
    composition, and host policy all match the original successful run.
    """

    checkpoint_family = _canonical_validation_checkpoint(checkpoint_id)

    digest = hashlib.sha256()
    for module in _validation_modules(checkpoint_family):
        digest.update(str(getattr(module, "__name__", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_digest(module).encode("ascii"))
        digest.update(b"\0")
    for name, value in sorted(
        (name, value)
        for name, value in os.environ.items()
        if name.startswith("MMM_")
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"=")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def validation_checkpoint_input(
    checkpoint_id: str,
    input_value: Mapping[str, Any],
) -> dict[str, Any]:
    scoped = dict(input_value)
    scoped["_mmm_validation_implementation"] = validation_implementation_fingerprint(
        checkpoint_id
    )
    return scoped


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _complete_source_receipt(value: Mapping[str, Any]) -> bool:
    checks_run = _nonnegative_int(value.get("checks_run"))
    findings = value.get("findings")
    if (
        value.get("status") != "PASS"
        or checks_run is None
        or checks_run <= 0
        or not isinstance(findings, list)
    ):
        return False
    for finding in findings:
        if not isinstance(finding, Mapping):
            return False
        severity = finding.get("severity")
        if not isinstance(severity, str):
            return False
        if severity.casefold() in {"error", "fatal"}:
            return False
    return True


def _complete_jar_receipt(value: Mapping[str, Any]) -> bool:
    """Allow resume only from a structurally complete passing JAR validation."""

    checks_run = _nonnegative_int(value.get("checks_run"))
    findings = value.get("findings")
    if (
        value.get("status") != "PASS"
        or checks_run is None
        or checks_run <= 0
        or not isinstance(findings, list)
    ):
        return False
    for finding in findings:
        if not isinstance(finding, Mapping):
            return False
        severity = finding.get("severity")
        if not isinstance(severity, str):
            return False
        if severity.casefold() in {"error", "fatal"}:
            return False
    return True


def _complete_jdt_receipt(value: Mapping[str, Any]) -> bool:
    """Accept only the canonical receipt emitted by JavaCoreService."""

    if value.get("schema_version") != "mmm/java-diagnostics-v3":
        return False
    if value.get("verification_backend") != "jdt_core":
        return False
    if value.get("complete") is not True or value.get("skipped") is not False:
        return False

    scope = value.get("verification_scope")
    if scope not in {"full", "incremental", "targeted"}:
        return False
    if not isinstance(value.get("project_root"), str) or not value["project_root"]:
        return False
    if not isinstance(value.get("project_revision"), Mapping):
        return False
    if not all(
        isinstance(value.get(key), str) and bool(value[key].strip())
        for key in ("model_id", "model_revision", "session_id")
    ):
        return False
    generation = value.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, (int, float)):
        return False

    diagnostics = value.get("diagnostics")
    error_count = _nonnegative_int(value.get("error_count"))
    warning_count = _nonnegative_int(value.get("warning_count"))
    if (
        not isinstance(diagnostics, Mapping)
        or error_count is None
        or warning_count is None
    ):
        return False

    observed_errors = 0
    observed_warnings = 0
    for uri, raw_items in diagnostics.items():
        if not isinstance(uri, str) or not uri or not isinstance(raw_items, list):
            return False
        for item in raw_items:
            if not isinstance(item, Mapping):
                return False
            severity = item.get("severity")
            if isinstance(severity, bool) or severity not in {1, 2, 3}:
                return False
            if severity == 1:
                observed_errors += 1
            elif severity == 2:
                observed_warnings += 1
    if observed_errors != error_count or observed_warnings != warning_count:
        return False

    if scope == "targeted":
        relative_files = value.get("relative_files")
        if (
            not isinstance(relative_files, list)
            or not relative_files
            or any(
                not isinstance(item, str) or not item.strip()
                for item in relative_files
            )
        ):
            return False
    elif "relative_files" in value:
        return False

    return True


def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    checkpoint_family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)
    if checkpoint_family == "validate-source":
        return _complete_source_receipt(value)
    if checkpoint_family == "validate-jdt":
        return _complete_jdt_receipt(value)
    if checkpoint_family == "validate-jar":
        return _complete_jar_receipt(value)
    return False


__all__ = [
    "cached_validation_is_reusable",
    "validation_checkpoint_input",
    "validation_implementation_fingerprint",
]
