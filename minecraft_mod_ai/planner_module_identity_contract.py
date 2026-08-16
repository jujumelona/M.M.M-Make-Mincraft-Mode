from __future__ import annotations

import re
from functools import wraps
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def install(planner_module: Any) -> None:
    """Keep planner module/dependency identities exact instead of auto-normalizing.

    ``ProductionModule.__post_init__`` retains a legacy normalization path for old
    saved objects. New structured planner output must not rely on it: changing
    ``Boss System`` to ``boss_system`` or rewriting dependency ids silently changes
    graph identity and can hide a planner reference error.
    """

    current = planner_module._module
    if getattr(current, "_mmm_exact_module_identity", False):
        return

    @wraps(current)
    def module(value: Any):
        if not isinstance(value, dict):
            return current(value)
        val = dict(value)
        raw_id = str(val.get("module_id") or val.get("id") or val.get("name") or "custom_module").strip()
        safe_id = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_id).strip("_").lower()
        if not safe_id or not safe_id[0].isalpha():
            safe_id = f"mod_{safe_id}".strip("_")
        val["module_id"] = safe_id[:64]

        dependencies = val.get("depends_on", [])
        if isinstance(dependencies, (list, tuple)):
            cleaned_deps = []
            for item in dependencies:
                d_str = str(item).strip()
                s_dep = re.sub(r"[^a-zA-Z0-9_]+", "_", d_str).strip("_").lower()
                if s_dep:
                    cleaned_deps.append(s_dep[:64])
            val["depends_on"] = cleaned_deps
        return current(val)

    module._mmm_exact_module_identity = True  # type: ignore[attr-defined]
    module.__wrapped__ = current  # type: ignore[attr-defined]
    planner_module._module = module


__all__ = ["install"]
