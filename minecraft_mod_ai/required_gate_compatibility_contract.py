from __future__ import annotations

from functools import wraps
from typing import Any


def _literal_command_passed(build_report: dict[str, Any] | None, name: str) -> bool:
    if not isinstance(build_report, dict):
        return False
    return any(
        isinstance(command, dict)
        and command.get("name") == name
        and command.get("exit_code") == 0
        and command.get("timed_out") is not True
        for command in build_report.get("commands", [])
    )


def install(orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator
    current = cls._command_receipt_passed
    if getattr(current, "_mmm_legacy_clean_build_compat", False):
        return

    @wraps(current)
    def command_receipt_passed(
        build_report: dict[str, Any] | None,
        name: str,
    ) -> bool:
        if name == "clean_build":
            # Current runtime calls this command "build"; old evidence receipts used
            # "clean_build". Both are valid for required-gate compatibility. The
            # quality build dimension is separately stricter and requires clean-room.
            return current(build_report, name) or _literal_command_passed(
                build_report, "clean_build"
            )
        return current(build_report, name)

    command_receipt_passed._mmm_legacy_clean_build_compat = True
    cls._command_receipt_passed = staticmethod(command_receipt_passed)
