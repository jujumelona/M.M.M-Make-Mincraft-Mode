from __future__ import annotations

"""Extract ordered, source-free procedural steps from verified work receipts."""

from collections.abc import Mapping, Sequence
from typing import Any

from .tool_transition_registry import reviewed_transition

_SCHEMA = "mmm/procedure-trace-v1"
_TERMINAL = {"PASS", "FAIL", "ERROR", "SUCCESS", "SUCCEEDED", "FAILED", "OK"}


def _status(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"PASS", "SUCCESS", "SUCCEEDED", "OK"}:
        return "PASS"
    if raw in {"FAIL", "FAILED", "ERROR"}:
        return "FAIL"
    return ""


def _effects(action: str) -> list[str]:
    spec = reviewed_transition(action)
    return sorted(spec.effects)[:8] if spec is not None else []


def _append(
    steps: list[dict[str, Any]],
    *,
    action: str,
    kind: str,
    status: str = "",
) -> None:
    action = action.strip()[:160]
    if not action or len(steps) >= 32:
        return
    record: dict[str, Any] = {
        "index": len(steps),
        "kind": kind,
        "action": action,
        "effects": _effects(action),
    }
    if status:
        record["status"] = status
    # Receipts sometimes repeat the same command in nested summaries. Preserve
    # ordering but remove exact adjacent duplicates so the trace stays compact.
    comparable = {key: value for key, value in record.items() if key != "index"}
    if steps:
        previous = {key: value for key, value in steps[-1].items() if key != "index"}
        if previous == comparable:
            return
    steps.append(record)


def _walk(value: Any, steps: list[dict[str, Any]], *, parent_key: str = "", depth: int = 0) -> None:
    if depth > 7 or len(steps) >= 32:
        return
    if isinstance(value, Mapping):
        # Model/tool action records use one of these explicit structural keys. Never
        # persist arguments, source contents, paths, prompts, error text or payloads.
        tool = value.get("tool")
        if isinstance(tool, str) and tool.strip():
            _append(steps, action=tool, kind="tool", status=_status(value.get("status")))
        action = value.get("action")
        if isinstance(action, str) and action.strip():
            _append(steps, action=action, kind="action", status=_status(value.get("status")))
        operation = value.get("operation")
        if isinstance(operation, str) and operation.strip():
            _append(steps, action=operation, kind="operation", status=_status(value.get("status")))

        # Gradle/runtime verifier command receipts conventionally carry `name` inside
        # a commands list. Names are reusable procedure structure; command arguments
        # and log paths are intentionally excluded.
        if parent_key == "commands":
            name = value.get("name")
            if isinstance(name, str) and name.strip():
                command_status = ""
                try:
                    exit_code = int(value.get("exit_code"))
                except (TypeError, ValueError):
                    exit_code = None
                if value.get("timed_out") is True or (exit_code is not None and exit_code != 0):
                    command_status = "FAIL"
                elif exit_code == 0:
                    command_status = "PASS"
                _append(steps, action=name, kind="verifier", status=command_status)

        # JDT is often represented as fields rather than a command object.
        if "jdt_status" in value or "jdt_error_count" in value:
            jdt = _status(value.get("jdt_status"))
            if not jdt:
                try:
                    count = int(value.get("jdt_error_count"))
                except (TypeError, ValueError):
                    count = None
                if count is not None:
                    jdt = "PASS" if count == 0 else "FAIL"
            _append(steps, action="jdt_diagnostics", kind="verifier", status=jdt)

        # Walk insertion order so ordered command/tool lists retain execution order.
        for key, child in value.items():
            if key in {"content", "source", "source_body", "source_code", "prompt", "arguments", "error", "log", "logs"}:
                continue
            _walk(child, steps, parent_key=str(key), depth=depth + 1)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value[:128]:
            _walk(child, steps, parent_key=parent_key, depth=depth + 1)
            if len(steps) >= 32:
                break


def extract_procedure(receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if isinstance(receipt, Mapping):
        _walk(receipt, steps)
    for index, step in enumerate(steps):
        step["index"] = index
    return {
        "schema_version": _SCHEMA,
        "ordered": True,
        "steps": steps,
    }


def sequence_actions(procedure: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(procedure, Mapping) or procedure.get("schema_version") != _SCHEMA:
        return ()
    raw = procedure.get("steps")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    actions: list[str] = []
    for step in raw[:32]:
        if not isinstance(step, Mapping):
            continue
        action = str(step.get("action", "")).strip()
        if action:
            actions.append(action)
    return tuple(actions)


__all__ = ["extract_procedure", "sequence_actions"]
