from __future__ import annotations

import re
from functools import wraps
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def install(planner_module: Any) -> None:
    """Keep planner module and dependency identities exact.

    New structured planner output must not rely on legacy object normalization:
    silently rewriting an identifier changes graph identity and can hide a planner
    reference error that the existing page-local repair path should correct.
    """
    current = planner_module._module
    if getattr(current, "_mmm_exact_module_identity", False):
        return

    @wraps(current)
    def module(value: Any):
        if not isinstance(value, dict):
            return current(value)
        raw_id = value.get("module_id") or value.get("id") or value.get("name")
        if not isinstance(raw_id, str) or not _ID.fullmatch(raw_id.strip()):
            raise planner_module.SpecValidationError(
                f"Production module id must already be lowercase snake_case: {raw_id!r}"
            )
        dependencies = value.get("depends_on", [])
        if isinstance(dependencies, (list, tuple)):
            invalid = [
                item
                for item in dependencies
                if not isinstance(item, str) or not _ID.fullmatch(item.strip())
            ]
            if invalid:
                raise planner_module.SpecValidationError(
                    f"Production module {raw_id.strip()} has invalid dependency ids: "
                    f"{invalid[:4]}"
                )
        return current(value)

    module._mmm_exact_module_identity = True  # type: ignore[attr-defined]
    module.__wrapped__ = current  # type: ignore[attr-defined]
    planner_module._module = module


__all__ = ["install"]
