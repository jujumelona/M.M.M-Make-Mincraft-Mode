from __future__ import annotations

"""Host support gate for Typed PlanIR production.

Typed PlanIR may only be produced for semantics that the deterministic backend can
actually bind. Unsupported platform behavior fails before Java generation; there
is no coder fallback.
"""

from collections.abc import Mapping
from typing import Any

from .typed_event_ir import infer_event_config, infer_event_type, is_mod_initialize_trigger
from .typed_plan_ir import typed_plan_reachable_function_ids

from .authored_structured_design import (
    active_concern_records,
    normalize_structured_sections,
)


_UNSUPPORTED_PLATFORM_SECTIONS = (
    "authority_and_network",
    "persistence",
    "resources_and_ui",
)

def _compact(value: Any) -> str:
    return " ".join(str(value or "").split())


def typed_plan_support_issues(
    structured_sections: Mapping[str, Any] | None,
    typed_plan_ir: Mapping[str, Any] | None = None,
    *,
    externally_covered_refs=(),
    runtime_owners=(),
) -> tuple[str, ...]:
    """Return deterministic unsupported-operation diagnostics."""

    normalized = normalize_structured_sections(structured_sections)
    from .request_runtime_records import owned_record_refs

    feature_owned = owned_record_refs(normalized, runtime_owners)
    issues: list[str] = []
    from .planning_detail_template import WORKSHEET_SECTIONS

    active_refs = {
        f"{section}.{concern}"
        for section in WORKSHEET_SECTIONS
        for concern, rows in active_concern_records(normalized, section).items()
        if rows
    }
    covered = {
        str(cover)
        for module in (
            typed_plan_ir.get("platform_modules", ())
            if isinstance(typed_plan_ir, Mapping)
            else ()
        )
        if isinstance(module, Mapping)
        for cover in module.get("covers", ())
        if isinstance(cover, str)
    }
    external_coverage = {
        str(ref).strip()
        for ref in externally_covered_refs
        if str(ref).strip()
    }
    unknown_external = external_coverage - active_refs
    if unknown_external:
        issues.extend(
            f"external.coverage_without_active_concern:{ref}"
            for ref in sorted(unknown_external)
        )
    reachable_functions = set(
        typed_plan_reachable_function_ids(typed_plan_ir)
        if isinstance(typed_plan_ir, Mapping)
        else ()
    )
    function_covered = {
        str(cover)
        for function in (
            typed_plan_ir.get("functions", ())
            if isinstance(typed_plan_ir, Mapping)
            else ()
        )
        if isinstance(function, Mapping)
        and str(function.get("id") or "") in reachable_functions
        for cover in function.get("covers", ())
        if isinstance(cover, str)
    }
    for function in (
        typed_plan_ir.get("functions", ())
        if isinstance(typed_plan_ir, Mapping)
        else ()
    ):
        if (
            isinstance(function, Mapping)
            and function.get("covers")
            and str(function.get("id") or "") not in reachable_functions
        ):
            issues.append(
                "function.unreachable:"
                + str(function.get("id") or "<missing>")
            )

    host_bound_coverage = covered | function_covered
    effective_coverage = host_bound_coverage | external_coverage
    phantom = sorted(host_bound_coverage - active_refs)
    issues.extend(
        f"host.coverage_without_active_concern:{ref}"
        for ref in phantom
    )

    for section in _UNSUPPORTED_PLATFORM_SECTIONS:
        active = active_concern_records(normalized, section)
        for concern, rows in active.items():
            ref = f"{section}.{concern}"
            if rows and ref not in effective_coverage:
                issues.append(ref)

    for section in (
        "behavior_contract",
        "algorithm",
        "failure_and_limits",
    ):
        active = active_concern_records(normalized, section)
        for concern, rows in active.items():
            ref = f"{section}.{concern}"
            remaining = [row for index, row in enumerate(rows)
                         if f"{ref}[{index}]" not in feature_owned]
            if remaining and ref not in function_covered:
                issues.append(ref)

    integration = active_concern_records(normalized, "integration")
    entry_points = tuple(integration.get("entry_points", ()))
    bindings_by_entry = {
        binding.get("entry_point_index"): binding
        for binding in (
            typed_plan_ir.get("event_bindings", ())
            if isinstance(typed_plan_ir, Mapping)
            else ()
        )
        if isinstance(binding, Mapping)
        and type(binding.get("entry_point_index")) is int
    }
    bound_entry_points = set(bindings_by_entry)

    for index, row in enumerate(entry_points):
        if f"integration.entry_points[{index}]" in feature_owned:
            if index in bound_entry_points:
                issues.append(f"integration.entry_points[{index}].duplicate_runtime_owner")
            continue
        trigger = row.get("trigger")
        if is_mod_initialize_trigger(trigger):
            if index in bound_entry_points:
                issues.append(
                    f"integration.entry_points[{index}].duplicate_mod_init_binding"
                )
            continue
        if index not in bound_entry_points:
            issues.append(
                "integration.entry_points"
                f"[{index}].trigger={_compact(trigger)!r}"
            )
            continue
        # A valid binding ID/signature alone cannot prove that it is the
        # *correct* entry point. Reject silently substituted lifecycle hooks,
        # and ensure commands retain the literal actually declared by design.
        binding = bindings_by_entry[index]
        expected_event = infer_event_type(trigger)
        actual_event = str(binding.get("event") or "")
        if expected_event is None or actual_event != expected_event:
            issues.append(
                f"integration.entry_points[{index}].event_mismatch:"
                f"declared={_compact(trigger)!r},bound={actual_event!r}"
            )
            continue
        if actual_event == "command":
            expected_config = infer_event_config("command", trigger)
            actual_config = binding.get("config")
            if (
                expected_config is None
                or not isinstance(actual_config, Mapping)
                or actual_config.get("literal") != expected_config["literal"]
            ):
                issues.append(
                    f"integration.entry_points[{index}].command_literal_mismatch"
                )

    # Additional player commands are host-generated from individual UI records,
    # not fictitious authored integration entry points. Validate their exact
    # deterministic identity and forwarded action so arbitrary extra bindings
    # cannot escape the integration contract.
    from .authored_ui_action_bindings import _mutating_ui_actions
    mutating_ui = bool(_mutating_ui_actions(normalized))
    ui_rows = active_concern_records(normalized, "resources_and_ui").get(
        "interactions", ()
    )
    from .authored_ui_action_bindings import action_record_identity
    ui_actions = tuple(dict.fromkeys(
        action_record_identity(row)
        for row in ui_rows
        if isinstance(row, Mapping) and action_record_identity(row)
    )) if mutating_ui else ()
    functions_by_id = {
        str(fn.get("id") or ""): fn
        for fn in (typed_plan_ir or {}).get("functions", ())
        if isinstance(fn, Mapping)
    }
    valid_entry_points = set(range(len(entry_points)))
    for index in sorted(bound_entry_points - valid_entry_points):
        ordinal = index - len(entry_points) + 1
        binding = bindings_by_entry[index]
        concrete = binding.get("config", {}).get("action")
        if concrete is not None:
            from .typed_action_contract import action_bindings
            contracts = action_bindings(typed_plan_ir or {})
            owned = {row["id"]: row for row in contracts}
            action_id = concrete.get("id")
            if (action_id in ui_actions and action_id in owned
                    and owned[action_id]["function"] == binding.get("function")
                    and binding.get("function") in reachable_functions):
                continue
            issues.append(f"event.binding_without_entry_point:{index}")
            continue
        function_id = f"uiAction{ordinal}_command"
        fn = functions_by_id.get(function_id, {})
        body = fn.get("body", ())
        first = body[0] if isinstance(body, (list, tuple)) and body else {}
        call = first.get("value") if isinstance(first, Mapping) else None
        args = call.get("args", ()) if isinstance(call, Mapping) else ()
        forwarded = args[0] if isinstance(args, (list, tuple)) and args else {}
        command_literal = str(binding.get("config", {}).get("literal") or "")
        if (
            1 <= ordinal <= len(ui_actions)
            and binding.get("event") == "command"
            and binding.get("function") == function_id
            and command_literal.startswith(f"mmm_action_{ordinal}")
            and str(fn.get("id") or "") in reachable_functions
            and first.get("op") == "return"
            and isinstance(call, Mapping)
            and call.get("op") == "call"
            and call.get("function") == "ui_action_dispatch"
            and isinstance(forwarded, Mapping)
            and forwarded.get("op") == "literal"
            and forwarded.get("type") == "string"
            and forwarded.get("value") == ui_actions[ordinal - 1]
        ):
            continue
        issues.append(f"event.binding_without_entry_point:{index}")

    return tuple(dict.fromkeys(issues))


def assert_typed_plan_host_support(
    structured_sections: Mapping[str, Any] | None,
    typed_plan_ir: Mapping[str, Any] | None = None,
    *,
    externally_covered_refs=(),
    runtime_owners=(),
) -> None:
    issues = typed_plan_support_issues(
        structured_sections,
        typed_plan_ir,
        externally_covered_refs=externally_covered_refs,
        runtime_owners=runtime_owners,
    )
    if issues:
        raise ValueError(
            "TYPED_PLAN_UNSUPPORTED_HOST_OPERATION: deterministic production "
            "has no host binding for "
            + ", ".join(issues)
            + ". Resolve this during planning; coder fallback is forbidden."
        )


__all__ = [
    "assert_typed_plan_host_support",
    "typed_plan_support_issues",
]
