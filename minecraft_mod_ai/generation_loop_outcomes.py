from __future__ import annotations

"""Semantic outcome classification for the generation tool loop."""

from collections.abc import Mapping, Sequence
from typing import Any

from .retrieval_progress import _stable_value
from .validation_diagnostic_contract import diagnostic_errors

MUTATION_ACT_TOOLS = frozenset({
    "apply_source_edit",
    "apply_source_patch",
    "apply_java_operations",
    "repair_project",
})
VERIFY_TOOLS = frozenset({
    "java_diagnostics",
    "jdt_diagnostics",
    "run_gametest",
})
_VERIFIER_UNAVAILABLE_STATUSES = frozenset({
    "UNAVAILABLE",
    "NOT_RUN",
    "TIMEOUT",
    "TIMED_OUT",
    "UNHEALTHY",
})
_VERIFIER_FAIL_STATUSES = frozenset({"FAIL", "FAILED", "ERROR", "INVALID"})
_VERIFIER_PASS_STATUSES = frozenset({
    "PASS",
    "PASSED",
    "OK",
    "SUCCESS",
    "SUCCEEDED",
    "AVAILABLE",
})


def verification_outcome(tool_name: str, payload: Mapping[str, Any]) -> str:
    if not bool(payload.get("ok")):
        code = str(payload.get("failure_code") or "").strip().upper()
        if code in {"VERIFIER_ARGUMENT_INVALID", "VERIFIER_TARGET_INVALID"}:
            return code
        return "UNAVAILABLE"

    result = payload.get("result")
    if not isinstance(result, Mapping):
        return "PASS"

    status = str(
        result.get("status")
        or result.get("state")
        or result.get("outcome")
        or ""
    ).strip().upper()
    if status in _VERIFIER_UNAVAILABLE_STATUSES:
        return "UNAVAILABLE"
    if result.get("available") is False:
        return "UNAVAILABLE"

    if tool_name in {"java_diagnostics", "jdt_diagnostics"}:
        errors = diagnostic_errors(result)
        unavailable_codes = {
            "JDT_DIAGNOSTICS_UNAVAILABLE",
            "JDT_WORKSPACE_NOT_READY",
        }
        if any(
            str(item.get("code") or "").strip().upper() in unavailable_codes
            for item in errors
        ):
            return "UNAVAILABLE"
        if errors:
            return "FAIL"
        return "FAIL" if status in _VERIFIER_FAIL_STATUSES else "PASS"

    if status in _VERIFIER_FAIL_STATUSES:
        return "FAIL"
    if status in _VERIFIER_PASS_STATUSES:
        return "PASS"

    for key in ("success", "ok"):
        if isinstance(result.get(key), bool):
            return "PASS" if result[key] else "FAIL"

    if "exit_code" in result:
        try:
            return "PASS" if int(result["exit_code"]) == 0 else "FAIL"
        except (TypeError, ValueError, OverflowError):
            return "UNAVAILABLE"
    return "PASS"


def fixed_point_tool_results(
    executed: Sequence[tuple[Any, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Use semantic outcomes, not volatile transport text, as loop identity."""

    stable: list[dict[str, Any]] = []
    for call, payload in executed:
        if call.name in VERIFY_TOOLS:
            stable.append(
                {
                    "name": call.name,
                    "verification_outcome": verification_outcome(
                        call.name,
                        payload,
                    ),
                }
            )
        else:
            stable.append(
                {
                    "name": call.name,
                    "ok": bool(payload.get("ok")),
                    "failure_code": payload.get("failure_code"),
                }
            )
    return stable


def fixed_point_tool_calls(calls: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip source payload bytes while preserving semantic action identity."""

    stable: list[dict[str, Any]] = []
    for call in calls:
        arguments = call.arguments if isinstance(call.arguments, Mapping) else {}
        item: dict[str, Any] = {"name": call.name}
        for key in (
            "operation",
            "path",
            "target_path",
            "file_path",
            "query",
            "capability",
        ):
            value = arguments.get(key)
            if value not in (None, "", [], {}, ()):
                item[key] = _stable_value(value)
        stable.append(item)
    return stable


def runtime_failure_code(tool_name: str, error: str) -> str:
    lowered = str(error or "").casefold()
    if tool_name in MUTATION_ACT_TOOLS:
        if "exact source-edit precondition failed" in lowered:
            return "MUTATION_STALE_PRECONDITION"
        if "target already exists" in lowered or "creation_conflict" in lowered:
            return "MUTATION_TARGET_CREATION_CONFLICT"
        if "source edit requires an existing regular file" in lowered:
            return "MUTATION_TARGET_UNBOUND"

    if tool_name in VERIFY_TOOLS:
        if any(
            marker in lowered
            for marker in (
                "no such file",
                "not found",
                "does not exist",
                "outside",
                "unsafe path",
            )
        ):
            return "VERIFIER_TARGET_INVALID"
        if any(marker in lowered for marker in ("argument", "schema", "invalid")):
            return "VERIFIER_ARGUMENT_INVALID"

    return "TOOL_RUNTIME_UNAVAILABLE"


__all__ = [
    "MUTATION_ACT_TOOLS",
    "VERIFY_TOOLS",
    "fixed_point_tool_calls",
    "fixed_point_tool_results",
    "runtime_failure_code",
    "verification_outcome",
]
