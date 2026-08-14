from __future__ import annotations

"""Install a pairwise active verifier used only for ambiguous repair candidates."""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .counterexample_verifier import discriminate


def _mode() -> str:
    value = os.environ.get("MMM_ACTIVE_REPAIR_VERIFIER", "auto").strip().casefold()
    return value if value in {"auto", "on", "off"} else "auto"


def _margin() -> float:
    try:
        value = float(os.environ.get("MMM_ACTIVE_REPAIR_SCORE_MARGIN", "80") or 80)
    except ValueError:
        value = 80.0
    return max(0.0, min(500.0, value))


def _ambiguous(
    evaluations: Sequence[tuple[float, int, Sequence[Mapping[str, Any]], Mapping[str, Any]]],
) -> bool:
    if len(evaluations) < 2:
        return False
    ordered = sorted(evaluations, key=lambda item: (-item[0], item[1]))
    left, right = ordered[0], ordered[1]
    if _mode() == "on":
        return True
    if _mode() == "off":
        return False
    gap = abs(float(left[0]) - float(right[0]))
    left_errors = left[3].get("jdt_error_count")
    right_errors = right[3].get("jdt_error_count")
    same_jdt = left_errors == right_errors
    same_status = str(left[3].get("jdt_status", "")) == str(right[3].get("jdt_status", ""))
    return gap <= _margin() or (same_jdt and same_status)


def install(optimization_module: Any) -> None:
    if getattr(optimization_module, "_mmm_active_candidate_discriminator", None) is not None:
        return

    def active_candidate_discriminator(
        root: Path | None,
        evaluations: Sequence[tuple[float, int, Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    ) -> list[tuple[float, int, Sequence[Mapping[str, Any]], Mapping[str, Any]]]:
        values = list(evaluations)
        if root is None or not root.is_dir() or not _ambiguous(values):
            return values
        ordered = sorted(values, key=lambda item: (-item[0], item[1]))
        left, right = discriminate(Path(root).resolve(), ordered[0], ordered[1])
        replacements = {left[1]: left, right[1]: right}
        result = [replacements.get(item[1], item) for item in values]
        print(
            "active counterexample verifier:",
            f"candidates={len(values)}",
            f"pair={ordered[0][1] + 1}/{ordered[1][1] + 1}",
            f"margin={abs(float(ordered[0][0]) - float(ordered[1][0])):.3f}",
            flush=True,
        )
        return result

    optimization_module._mmm_active_candidate_discriminator = active_candidate_discriminator


__all__ = ["install"]
