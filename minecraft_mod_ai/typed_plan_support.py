from __future__ import annotations

"""Host support gate for Typed PlanIR production.

Typed PlanIR may only be produced for semantics that the deterministic backend can
actually bind. Unsupported platform behavior fails before Java generation; there
is no coder fallback.
"""

import re
from collections.abc import Mapping
from typing import Any

from .authored_structured_design import (
    active_concern_records,
    normalize_structured_sections,
)


_UNSUPPORTED_PLATFORM_SECTIONS = (
    "authority_and_network",
    "persistence",
    "resources_and_ui",
)

_LIFECYCLE_TRIGGER = re.compile(
    r"^(?:"
    r"mod[ _-]?(?:init|initialize|initialization|startup)|"
    r"oninitialize|initialize|initialization|startup|server[ _-]?startup|"
    r"모드[ _-]?(?:초기화|시작)|초기화|시작"
    r")$",
    re.IGNORECASE,
)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _lifecycle_trigger(value: Any) -> bool:
    normalized = re.sub(r"[\s:/\\]+", "_", _compact(value)).strip("_")
    return bool(_LIFECYCLE_TRIGGER.fullmatch(normalized))


def typed_plan_support_issues(
    structured_sections: Mapping[str, Any] | None,
    typed_plan_ir: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return deterministic unsupported-operation diagnostics."""

    normalized = normalize_structured_sections(structured_sections)
    issues: list[str] = []
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

    for section in _UNSUPPORTED_PLATFORM_SECTIONS:
        active = active_concern_records(normalized, section)
        for concern, rows in active.items():
            ref = f"{section}.{concern}"
            if rows and ref not in covered:
                issues.append(ref)

    integration = active_concern_records(normalized, "integration")
    for index, row in enumerate(integration.get("entry_points", ())):
        trigger = row.get("trigger")
        if (
            not _lifecycle_trigger(trigger)
            and "integration.entry_points" not in covered
        ):
            issues.append(
                "integration.entry_points"
                f"[{index}].trigger={_compact(trigger)!r}"
            )

    return tuple(dict.fromkeys(issues))


def assert_typed_plan_host_support(
    structured_sections: Mapping[str, Any] | None,
    typed_plan_ir: Mapping[str, Any] | None = None,
) -> None:
    issues = typed_plan_support_issues(
        structured_sections,
        typed_plan_ir,
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
