from __future__ import annotations

"""Small verifier-repair admission helpers kept out of the main tool loop."""

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any


def target_scoped_verifier_files(
    forced_verifier: str | None,
    context: Any,
) -> tuple[str, ...]:
    if forced_verifier != "java_diagnostics" or context is None:
        return ()
    path = str(getattr(context, "target_path", "") or "").replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    if not path or not path.casefold().endswith(".java"):
        return ()
    return (path,)


def reject_noop_repair(
    turn: Any,
    *,
    state: Any,
    binder: Callable[[Any, Any], Any],
) -> Any:
    """Reject a byte-identical verifier repair before runtime mutation."""

    if str(getattr(state, "validation_status", "") or "") != "FAIL":
        return turn
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    if len(calls) != 1 or str(getattr(calls[0], "name", "") or "") != "apply_source_edit":
        return turn
    bound = binder(calls[0], state)
    arguments = getattr(bound, "arguments", None)
    if not isinstance(arguments, Mapping):
        return turn
    old = arguments.get("old")
    new = arguments.get("new")
    if not isinstance(old, str) or not isinstance(new, str) or old != new:
        return turn

    raw = json.dumps(
        {"new": new},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "failure_code": "REPAIR_ATOMIC_NOOP",
        "original_tool": "apply_source_edit",
        "raw_arguments": raw,
        "error": (
            "verifier repair replacement is byte-identical to the host-selected "
            "old span; do not echo the old text. Emit a materially different local "
            "replacement that addresses the active diagnostic."
        ),
    }
    rejected = replace(
        bound,
        name="__mmm_rejected_tool_call__",
        arguments=payload,
        raw_arguments=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    return replace(turn, tool_calls=(rejected,))
