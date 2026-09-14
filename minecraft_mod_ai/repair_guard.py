"""Defense-in-depth wrapper that keeps infrastructure failures out of coder repair."""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .repair_engine import RepairEngine as _BaseRepairEngine
from .repairability import source_repair_block_reason

_PRELOADED_EVIDENCE: ContextVar[tuple[Path, bool, dict[str, Any]] | None] = ContextVar(
    "mmm_repair_guard_preloaded_evidence", default=None
)


def _evidence_block_reason(evidence: dict[str, Any]) -> str | None:
    build = evidence.get("build")
    diagnostics = evidence.get("diagnostics")
    return source_repair_block_reason(
        build=build if isinstance(build, dict) else None,
        diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
    )


def _blocked_result(evidence: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": "mmm/repair-result-v2",
        "status": "FAIL",
        "attempts": 0,
        "stop_reason": "non_source_repairable",
        "repairable": False,
        "reason": reason,
        "evidence": evidence,
        "patch_receipts": [],
    }


def _passed_result(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "mmm/repair-result-v2",
        "status": "PASS",
        "attempts": 0,
        "evidence": evidence,
        "patch_receipts": [],
    }


class RepairEngine(_BaseRepairEngine):
    """Repair engine variant that never asks the coder to patch non-source failures."""

    def repair(
        self,
        project_root: str | Path,
        *,
        run_gametest: bool = True,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        evidence = super()._evidence(root, run_gametest=run_gametest)
        if evidence.get("passed") is True:
            return _passed_result(evidence)

        reason = _evidence_block_reason(evidence)
        if reason is not None:
            print(
                f"  [!] Source repair suppressed for non-source failure: {reason}",
                flush=True,
            )
            return _blocked_result(evidence, reason)

        token = _PRELOADED_EVIDENCE.set((root, run_gametest, evidence))
        try:
            return super().repair(
                root,
                run_gametest=run_gametest,
                max_attempts=max_attempts,
            )
        finally:
            _PRELOADED_EVIDENCE.reset(token)

    def _evidence(
        self,
        root: Path,
        *,
        run_gametest: bool,
    ) -> dict[str, Any]:
        preloaded = _PRELOADED_EVIDENCE.get()
        if preloaded is not None and preloaded[0] == root and preloaded[1] == run_gametest:
            _PRELOADED_EVIDENCE.set(None)
            return preloaded[2]
        return super()._evidence(root, run_gametest=run_gametest)

    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reason = _evidence_block_reason(evidence)
        if reason is not None:
            print(
                f"  [!] Source repair suppressed for non-source failure: {reason}",
                flush=True,
            )
            return []
        return super()._request_patch(evidence, context)
