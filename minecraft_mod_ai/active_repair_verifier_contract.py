from __future__ import annotations

"""Install a pairwise active verifier used only for ambiguous repair candidates."""

import os
from functools import wraps
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


def _evidence_seed(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only failure-oracle structure needed to generate a regression test."""

    diagnostics_root = evidence.get("diagnostics")
    diagnostics_root = diagnostics_root if isinstance(diagnostics_root, Mapping) else {}
    raw = diagnostics_root.get("diagnostics", ())
    diagnostics: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw[:16]:
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code", "")).strip()[:120]
            message = " ".join(str(item.get("message", "")).split())[:240]
            try:
                severity = int(item.get("severity", 0) or 0)
            except (TypeError, ValueError):
                severity = 0
            if code or message:
                diagnostics.append({"code": code, "message": message, "severity": severity})

    build = evidence.get("build")
    build = build if isinstance(build, Mapping) else {}
    failing_commands: list[str] = []
    commands = build.get("commands", ())
    if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
        for command in commands[:16]:
            if not isinstance(command, Mapping):
                continue
            try:
                exit_code = int(command.get("exit_code"))
            except (TypeError, ValueError):
                exit_code = None
            if command.get("timed_out") is True or (exit_code is not None and exit_code != 0):
                name = str(command.get("name", "")).strip()
                if name:
                    failing_commands.append(name[:120])
    return {
        "schema_version": "mmm/counterexample-seed-v1",
        "diagnostics": diagnostics,
        "failing_commands": sorted(set(failing_commands))[:16],
        "build_status": str(build.get("status", ""))[:40],
    }


def _install_seed_wrapper(optimization_module: Any) -> None:
    current = optimization_module._verify_repair_candidate
    if getattr(current, "_mmm_counterexample_seed", False):
        return

    @wraps(current)
    def verify_with_seed(self: Any, root: Path | None, operations: Any, evidence: Mapping[str, Any]):
        score, verifier = current(self, root, operations, evidence)
        updated = dict(verifier) if isinstance(verifier, Mapping) else {"verifier": verifier}
        updated["counterexample_seed"] = _evidence_seed(evidence)
        return score, updated

    verify_with_seed._mmm_counterexample_seed = True  # type: ignore[attr-defined]
    verify_with_seed.__wrapped__ = current  # type: ignore[attr-defined]
    optimization_module._verify_repair_candidate = verify_with_seed


def install(optimization_module: Any) -> None:
    _install_seed_wrapper(optimization_module)
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


__all__ = ["_ambiguous", "install"]
