from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})


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

    from . import complete_orchestrator, runtime_bootstrap

    common: list[Any] = [
        sys.modules[__name__],
        runtime_bootstrap,
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
    else:
        from . import (
            java_lsp,
            java_lsp_process_safety_contract,
            research_validation_fingerprint_performance,
            validation_diagnostic_contract,
            validation_execution_contract,
        )

        common.extend(
            (
                java_lsp,
                java_lsp_process_safety_contract,
                validation_diagnostic_contract,
                validation_execution_contract,
                research_validation_fingerprint_performance,
            )
        )
    return tuple(common)


def validation_implementation_fingerprint(checkpoint_id: str) -> str:
    """Hash the complete active validation implementation and MMM host policy.

    Validation checkpoints are reusable only when their generated inputs, validation
    implementation (including runtime-installed validation contracts), bootstrap
    composition, and host policy all match the original successful run.
    """

    if checkpoint_id not in _VALIDATION_CHECKPOINTS:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")

    digest = hashlib.sha256()
    for module in _validation_modules(checkpoint_id):
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


def _diagnostic_sort_key(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _complete_jdt_receipt(value: Mapping[str, Any]) -> bool:
    if value.get("schema_version") != "mmm/java-diagnostics-v2":
        return False

    raw_diagnostics = value.get("diagnostics")
    diagnostics_by_uri = value.get("diagnostics_by_uri")
    legacy_errors: list[dict[str, Any]] | None = None
    if isinstance(raw_diagnostics, Mapping):
        diagnostics = raw_diagnostics
        if diagnostics_by_uri is not None and diagnostics_by_uri != raw_diagnostics:
            return False
    elif isinstance(raw_diagnostics, list) and isinstance(diagnostics_by_uri, Mapping):
        if any(not isinstance(item, Mapping) for item in raw_diagnostics):
            return False
        diagnostics = diagnostics_by_uri
        legacy_errors = [dict(item) for item in raw_diagnostics]
    else:
        return False

    pages = value.get("pages")
    files_opened = _nonnegative_int(value.get("files_opened"))
    page_count = _nonnegative_int(value.get("page_count"))
    error_count = _nonnegative_int(value.get("error_count"))
    warning_count = _nonnegative_int(value.get("warning_count"))
    if (
        not isinstance(pages, list)
        or files_opened is None
        or page_count is None
        or error_count is None
        or warning_count is None
        or page_count != len(pages)
        or len(diagnostics) != files_opened
    ):
        return False

    observed_errors = 0
    observed_warnings = 0
    expected_legacy_errors: list[dict[str, Any]] = []
    for uri, raw_items in diagnostics.items():
        if not isinstance(uri, str) or not uri or not isinstance(raw_items, list):
            return False
        for item in raw_items:
            if not isinstance(item, Mapping):
                return False
            severity = item.get("severity", 1)
            if isinstance(severity, bool) or not isinstance(severity, int):
                return False
            if severity == 1:
                observed_errors += 1
                expected_legacy_errors.append(dict(item))
            elif severity == 2:
                observed_warnings += 1
    if observed_errors != error_count or observed_warnings != warning_count:
        return False
    if legacy_errors is not None and sorted(
        legacy_errors, key=_diagnostic_sort_key
    ) != sorted(expected_legacy_errors, key=_diagnostic_sort_key):
        return False

    page_files = 0
    page_diagnostics = 0
    page_errors = 0
    page_warnings = 0
    for page in pages:
        if not isinstance(page, Mapping):
            return False
        file_count = _nonnegative_int(page.get("file_count"))
        diagnostic_uri_count = _nonnegative_int(page.get("diagnostic_uri_count"))
        errors = _nonnegative_int(page.get("error_count"))
        warnings = _nonnegative_int(page.get("warning_count"))
        if (
            file_count is None
            or diagnostic_uri_count is None
            or errors is None
            or warnings is None
            or diagnostic_uri_count != file_count
        ):
            return False
        page_files += file_count
        page_diagnostics += diagnostic_uri_count
        page_errors += errors
        page_warnings += warnings
    return (
        page_files == files_opened
        and page_diagnostics == files_opened
        and page_errors == error_count
        and page_warnings == warning_count
    )


def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if checkpoint_id == "validate-source":
        return _complete_source_receipt(value)
    if checkpoint_id == "validate-jdt":
        return _complete_jdt_receipt(value)
    return False


__all__ = [
    "cached_validation_is_reusable",
    "validation_checkpoint_input",
    "validation_implementation_fingerprint",
]
