from __future__ import annotations

"""Host-owned convergence frontier for verifier repair."""

from collections.abc import Mapping, Sequence
from typing import Any

from .verifier_repair_window import select_verifier_repair_window


def target_scoped_verifier_files(
    forced_verifier: str | None,
    mutation_context: Any,
) -> tuple[str, ...]:
    if forced_verifier != "java_diagnostics" or mutation_context is None:
        return ()
    if bool(getattr(mutation_context, "is_new_file", False)):
        return ()
    path = str(getattr(mutation_context, "target_path", "") or "").replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    if not path or not path.casefold().endswith(".java"):
        return ()
    return (path,)


def select_repair_window_from_frontier(
    source: str,
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    cursor: int,
    start_line: int | None,
    end_line: int | None,
) -> dict[str, Any] | None:
    start = max(0, int(cursor or 0))
    for index in range(start, len(diagnostics)):
        window = select_verifier_repair_window(
            source,
            (diagnostics[index],),
            start_line=start_line,
            end_line=end_line,
        )
        if window is None:
            continue
        return {
            **window,
            "diagnostic_index": index,
            "diagnostic_frontier_index": index,
        }
    return None


def bound_repair_call_is_noop(call: Any) -> bool:
    if str(getattr(call, "name", "") or "") != "apply_source_edit":
        return False
    arguments = getattr(call, "arguments", None)
    if not isinstance(arguments, Mapping):
        return False
    return (
        str(arguments.get("operation") or "").strip().casefold() == "replace_exact"
        and isinstance(arguments.get("old"), str)
        and arguments.get("new") == arguments.get("old")
    )


def repair_rejection_payloads_are_same_route(
    payloads: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(payloads) and all(
        str(payload.get("original_tool") or payload.get("rejected_name") or "")
        == "apply_source_edit"
        for payload in payloads
    )
