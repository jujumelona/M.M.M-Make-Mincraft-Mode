from __future__ import annotations

"""Strict production boundary for canonical structured state models.

Planning owns state semantics. Production never renames identifiers, repairs
expressions, drops mutations, invents defaults, or otherwise reinterprets the
canonical state authority. It only verifies the persisted contract and lowers it
through the host compiler.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import re
from typing import Any

from .authored_plan import AuthoredPlan
from .planning_detail_slots import DETAIL_RECORDS
from .structured_state_runtime import (
    render_state_model_concern,
    validate_structured_state_section,
)

_STATE_CONCERNS = tuple(DETAIL_RECORDS["state_model"])


def normalize_structured_state_section(
    section: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a structural copy of canonical typed state authority, or fail closed."""

    if not isinstance(section, Mapping):
        raise ValueError("PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED")

    specification = section.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError(
            "PRODUCTION_STATE_SPECIFICATION_REQUIRED: canonical state authority "
            "must contain a specification object"
        )

    result_specification: dict[str, Any] = {}
    for concern in _STATE_CONCERNS:
        rows = specification.get(concern, ())
        if rows is None:
            rows = ()
        if not isinstance(rows, Sequence) or isinstance(
            rows, (str, bytes, bytearray)
        ):
            raise ValueError(
                "PRODUCTION_STATE_CONCERN_INVALID: "
                f"state_model.{concern} must be an array"
            )

        copied_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(
                    "PRODUCTION_STATE_RECORD_INVALID: "
                    f"state_model.{concern}[{index}] must be an object"
                )
            copied_rows.append(deepcopy(dict(row)))
        result_specification[concern] = copied_rows

    raw_inapplicable = specification.get("inapplicable_concerns", ())
    if raw_inapplicable is None:
        raw_inapplicable = ()
    if not isinstance(raw_inapplicable, Sequence) or isinstance(
        raw_inapplicable, (str, bytes, bytearray)
    ):
        raise ValueError(
            "PRODUCTION_STATE_INAPPLICABLE_INVALID: "
            "state_model.inapplicable_concerns must be an array"
        )
    inapplicable: list[dict[str, Any]] = []
    for index, row in enumerate(raw_inapplicable):
        if not isinstance(row, Mapping):
            raise ValueError(
                "PRODUCTION_STATE_INAPPLICABLE_INVALID: "
                f"state_model.inapplicable_concerns[{index}] must be an object"
            )
        inapplicable.append(deepcopy(dict(row)))
    result_specification["inapplicable_concerns"] = inapplicable

    raw_evidence = section.get("constraint_evidence_refs", ())
    if raw_evidence is None:
        raw_evidence = ()
    if not isinstance(raw_evidence, Sequence) or isinstance(
        raw_evidence, (str, bytes, bytearray)
    ):
        raise ValueError(
            "PRODUCTION_STATE_EVIDENCE_INVALID: "
            "constraint_evidence_refs must be an array"
        )

    normalized_section = {
        "specification": result_specification,
        "constraint_evidence_refs": [
            str(value)
            for value in raw_evidence
            if str(value).strip()
        ],
    }
    validate_structured_state_section(result_specification)
    return normalized_section


def render_production_state_java(
    section: Mapping[str, Any],
    *,
    package_name: str,
    symbol: str = "AuthoredStateModel",
) -> str:
    """Render one complete state owner class from canonical structured authority."""

    if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", symbol):
        raise ValueError(f"PRODUCTION_STATE_SYMBOL_INVALID: {symbol!r}")
    package = str(package_name or "").strip()
    if package and re.fullmatch(
        r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*",
        package,
    ) is None:
        raise ValueError(f"PRODUCTION_STATE_PACKAGE_INVALID: {package!r}")

    normalized = normalize_structured_state_section(section)
    specification = normalized["specification"]
    obligations: list[str] = []
    active: list[str] = []
    for concern in _STATE_CONCERNS:
        rows = specification.get(concern)
        if not isinstance(rows, list) or not rows:
            continue
        active.append(concern)
        obligations.append(
            json.dumps(
                {
                    "instruction": json.dumps(
                        {"concern": concern},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "structured_records": rows,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    if not active:
        raise ValueError(
            "PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED: "
            "state operations require at least one canonical state concern record"
        )

    task = {"implementation_obligations": obligations}
    members: list[str] = []
    include_runtime = True
    for concern in active:
        rendered = render_state_model_concern(
            task,
            concern,
            include_runtime=include_runtime,
        )
        if not rendered:
            continue
        members.append(rendered)
        include_runtime = False

    if not members:
        raise ValueError("PRODUCTION_STATE_HOST_COMPILER_EMPTY")

    body = "\n\n".join(members)
    indented = "\n".join(
        ("    " + line if line else "")
        for line in body.splitlines()
    )
    prefix = f"package {package};\n\n" if package else ""
    return (
        prefix
        + "// MMM:TYPED_PLAN_STATE_OWNER\n"
        + f"public final class {symbol} {{\n"
        + f"    private {symbol}() {{}}\n\n"
        + indented
        + "\n}\n"
    )


def compile_production_state_section(plan: AuthoredPlan) -> dict[str, Any]:
    """Validate and freeze canonical structured state for deterministic production."""

    structured = plan.structured_sections
    if not isinstance(structured, Mapping):
        raise ValueError("PRODUCTION_STATE_STRUCTURED_AUTHORITY_REQUIRED")
    raw_state = structured.get("state_model")
    if not isinstance(raw_state, Mapping):
        return {}
    return normalize_structured_state_section(raw_state)


__all__ = [
    "compile_production_state_section",
    "normalize_structured_state_section",
    "render_production_state_java",
]
