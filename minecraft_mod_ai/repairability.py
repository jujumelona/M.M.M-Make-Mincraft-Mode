"""Single policy for deciding whether verifier evidence may mutate source files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_NON_SOURCE_STATUSES = {
    "UNAVAILABLE",
    "BLOCKED",
    "SKIP",
    "SKIPPED",
    "CANCELLED",
    "CANCELED",
}
_NON_SOURCE_FAILURE_CLASSES = {
    "environment",
    "infrastructure",
    "toolchain",
    "availability",
    "verifier",
    "workspace",
    "validation",
    "validation_consistency",
}
_NON_SOURCE_CODES = {
    "JAVA_TOOLCHAIN_UNAVAILABLE",
    "JDT_DIAGNOSTICS_UNAVAILABLE",
    "JDT_WORKSPACE_NOT_READY",
    "JDT_OWNER_RESOLVE_TIMEOUT",
    "VERIFIER_UNAVAILABLE",
    "VALIDATION_INPUTS_CHANGED",
    "VALIDATION_INPUT_CHANGED",
    "PROJECT_INPUTS_CHANGED",
}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _explicitly_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in _FALSE_VALUES
    return False


def _normalized_text(value: Any) -> str:
    parts: list[str] = []
    for mapping in _walk_mappings(value):
        for key in (
            "code",
            "error_code",
            "error",
            "message",
            "reason",
            "failure_class",
            "stop_reason",
        ):
            item = mapping.get(key)
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
    return " ".join(" ".join(parts).casefold().split())


def _non_source_code(value: Any) -> str | None:
    for mapping in _walk_mappings(value):
        for key in ("code", "error_code"):
            raw = mapping.get(key)
            if not isinstance(raw, str):
                continue
            code = raw.strip().upper()
            if code in _NON_SOURCE_CODES:
                return code
    return None


def _build_status_reason(build: Mapping[str, Any]) -> str | None:
    status = str(build.get("status") or "").strip().upper()
    if status == "PASS":
        return "build_already_passed"
    if status in _NON_SOURCE_STATUSES:
        return f"build_status_{status.casefold()}"
    if status != "FAIL":
        return f"non_source_build_status_{status.casefold() or 'missing'}"
    return None


def _build_structure_reason(build: Mapping[str, Any]) -> str | None:
    if _explicitly_false(build.get("repairable")):
        return "build_marked_non_repairable"
    failure_class = str(build.get("failure_class") or "").strip().casefold()
    if failure_class in _NON_SOURCE_FAILURE_CLASSES:
        return f"failure_class_{failure_class}"
    code = _non_source_code(build)
    if code is not None:
        return f"non_source_code_{code.casefold()}"
    return None


def _command_timeout_reason(build: Mapping[str, Any]) -> str | None:
    commands = build.get("commands")
    if not isinstance(commands, (list, tuple)):
        return None
    for command in commands:
        if isinstance(command, Mapping) and bool(command.get("timed_out")):
            return "build_command_timed_out"
    return None


def _diagnostics_reason(diagnostics: Mapping[str, Any]) -> str | None:
    for mapping in _walk_mappings(diagnostics):
        status = str(mapping.get("status") or "").strip().upper()
        if status in _NON_SOURCE_STATUSES:
            return f"diagnostics_status_{status.casefold()}"
        if _explicitly_false(mapping.get("repairable")):
            return "diagnostics_marked_non_repairable"
        failure_class = str(mapping.get("failure_class") or "").strip().casefold()
        if failure_class in _NON_SOURCE_FAILURE_CLASSES:
            return f"diagnostics_failure_class_{failure_class}"
    code = _non_source_code(diagnostics)
    if code is not None:
        return f"non_source_code_{code.casefold()}"
    return None


def _legacy_text_reason(text: str) -> str | None:
    if "release version" in text and "not supported" in text:
        return "java_release_not_supported"
    if "java" in text and "toolchain unavailable" in text:
        return "java_toolchain_unavailable"
    if "owner resolve timed out" in text:
        return "jdt_owner_resolve_timeout"
    if "jdt diagnostics are unavailable" in text:
        return "jdt_diagnostics_unavailable"
    if "jdt" in text and "workspace is not ready" in text:
        return "jdt_workspace_not_ready"
    if "project inputs changed during validation" in text:
        return "validation_inputs_changed"
    if "result is not certifiable" in text:
        return "validation_not_certifiable"
    if "timeouterror" in text or "timed out" in text:
        return "infrastructure_timeout"
    return None


def source_repair_block_reason(
    *,
    build: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None = None,
) -> str | None:
    """Return why source mutation is forbidden, or ``None`` for a source failure."""

    if not isinstance(build, Mapping):
        return "invalid_build_receipt"

    for check in (_build_status_reason, _build_structure_reason, _command_timeout_reason):
        reason = check(build)
        if reason is not None:
            return reason

    if diagnostics is not None:
        reason = _diagnostics_reason(diagnostics)
        if reason is not None:
            return reason

    text = _normalized_text({"build": build, "diagnostics": diagnostics or {}})
    return _legacy_text_reason(text)


def source_repair_allowed(
    *,
    build: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None = None,
) -> bool:
    return source_repair_block_reason(build=build, diagnostics=diagnostics) is None
