"""Deterministic host lowering for canonical structured state records."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

class CustomModuleGenerationError(RuntimeError):
    pass
from .structured_state_runtime import render_state_model_concern


def prepare_state_concern(
    task: Mapping[str, Any],
    concern: str,
    *,
    include_runtime: bool,
) -> str | None:
    """Compile one state concern; never delegate Java work to the coder."""

    try:
        return render_state_model_concern(
            task,
            concern,
            include_runtime=include_runtime,
        )
    except ValueError as exc:
        raise CustomModuleGenerationError(
            "STRUCTURED_STATE_HOST_DSL_REQUIRED: planning admitted a state record "
            "outside the host compiler DSL. Repair the structured planning record "
            "instead of invoking a Java coder. "
            + str(exc)
        ) from exc
