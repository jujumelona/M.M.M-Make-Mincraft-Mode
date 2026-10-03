"""Lower authored state fields without confusing design text with executable DSL."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .custom_module_errors import CustomModuleGenerationError
from .java_region_parser import class_body_member_contracts
from .structured_state_runtime import PUBLIC_API, render_state_model_concern


def prepare_state_concern(task: Mapping[str, Any], concern: str, *, include_runtime: bool):
    """Compile state concerns deterministically; production never asks the coder for Java."""

    try:
        members = render_state_model_concern(
            task,
            concern,
            include_runtime=include_runtime,
        )
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "STRUCTURED_STATE_HOST_DSL_REQUIRED: planning admitted a state record "
            "outside the host compiler DSL. State-model Java fallback is disabled; "
            "repair the structured planning record instead. "
            + str(exc)
        ) from exc

    return members, {
        "work": [],
        "runtime_api": list(PUBLIC_API),
        "rules": [
            "State-model source is host compiled from canonical structured records.",
            "The coder must never author Java declarations, signatures, bodies, or repairs for state_model.",
        ],
    }


def validate_state_helpers(source: str, contract: Mapping[str, Any]) -> None:
    """Require executable helper declarations before host registration is assembled."""
    methods = class_body_member_contracts(source)
    for work in contract["work"]:
        matches = [row for row in methods if row.get("kind") == "method"
                   and row.get("symbol") == work["symbol"]]
        expected_parameters = work.get("parameters") or [
            {"type": "java.util.Map<String, Object>", "name": "context"},
        ]
        if not any(
            row.get("visibility") == "private" and row.get("static")
            and row.get("return_type") == work["return_type"]
            and row.get("parameters") == expected_parameters
            for row in matches
        ):
            raise CustomModuleGenerationError(
                "ATOMIC_CONCERN_RESPONSE_INVALID: missing authored state implementation: "
                + work["declaration"]
            )
