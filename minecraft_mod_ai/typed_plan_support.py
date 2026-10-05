from __future__ import annotations

"""Host support gate for Typed PlanIR production.

Typed PlanIR may only be produced for semantics that the deterministic backend can
actually bind. Unsupported platform behavior fails before Java generation; there
is no coder fallback.
"""

from collections.abc import Mapping
from typing import Any

from .typed_event_ir import is_mod_initialize_trigger
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
) -> tuple[str, ...]:
    """Return deterministic unsupported-operation diagnostics."""

    normalized = normalize_structured_sections(structured_sections)
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
            if rows and ref not in function_covered:
                issues.append(ref)

    integration = active_concern_records(normalized, "integration")
    entry_points = tuple(integration.get("entry_points", ()))
    bound_entry_points = {
        binding.get("entry_point_index")
        for binding in (
            typed_plan_ir.get("event_bindings", ())
            if isinstance(typed_plan_ir, Mapping)
            else ()
        )
        if isinstance(binding, Mapping)
        and type(binding.get("entry_point_index")) is int
    }

    for index, row in enumerate(entry_points):
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

    valid_entry_points = set(range(len(entry_points)))
    for index in sorted(bound_entry_points - valid_entry_points):
        issues.append(
            f"event.binding_without_entry_point:{index}"
        )

    return tuple(dict.fromkeys(issues))


def assert_typed_plan_host_support(
    structured_sections: Mapping[str, Any] | None,
    typed_plan_ir: Mapping[str, Any] | None = None,
    *,
    externally_covered_refs=(),
) -> None:
    issues = typed_plan_support_issues(
        structured_sections,
        typed_plan_ir,
        externally_covered_refs=externally_covered_refs,
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
