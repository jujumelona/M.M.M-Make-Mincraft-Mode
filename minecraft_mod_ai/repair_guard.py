"""Defense-in-depth wrapper that keeps infrastructure failures out of coder repair."""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .repair_engine import RepairEngine as _BaseRepairEngine
from .repairability import source_repair_block_reason
from .runner import BuildRunnerError

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


def _initial_compile_evidence(build: dict[str, Any]) -> dict[str, Any]:
    """Convert the already executed target compile into first repair evidence."""

    return {
        "passed": build.get("status") == "PASS",
        "diagnostics": {
            "schema_version": "mmm/java-diagnostics-v3",
            "status": "DEFERRED_TO_POST_BUILD",
            "available": False,
            "complete": False,
            "diagnostics": {},
        },
        "build": dict(build),
    }


class RepairEngine(_BaseRepairEngine):
    """Repair engine variant that never asks the coder to patch non-source failures."""

    def repair(
        self,
        project_root: str | Path,
        *,
        run_gametest: bool = True,
        max_attempts: int | None = None,
        initial_build: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        root = Path(project_root).expanduser().resolve()
        evidence = (
            _initial_compile_evidence(initial_build)
            if isinstance(initial_build, dict)
            else super()._evidence(root, run_gametest=run_gametest)
        )
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
        """Use the real target compiler as the only repair oracle.

        Repair must not inherit auxiliary JDT wrappers from the base RepairEngine.
        The first failed Gradle receipt is preloaded by run_build_with_repair; every
        subsequent repair iteration rebuilds the exact target and feeds that compiler
        result back to the coder. JDT remains a post-build validation service only.
        """

        preloaded = _PRELOADED_EVIDENCE.get()
        if preloaded is not None and preloaded[0] == root and preloaded[1] == run_gametest:
            _PRELOADED_EVIDENCE.set(None)
            return preloaded[2]

        try:
            build = self.runner_factory(self.gradle_cache).build(
                root,
                run_gametest=run_gametest,
            ).to_dict()
        except (BuildRunnerError, OSError, TimeoutError) as exc:
            build = {
                "status": "UNAVAILABLE",
                "failure_class": "infrastructure",
                "repairable": False,
                "error_code": "TARGET_COMPILE_UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "commands": [],
            }
        return _initial_compile_evidence(build)

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
