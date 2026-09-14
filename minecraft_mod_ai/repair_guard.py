"""Defense-in-depth wrapper that keeps infrastructure failures out of coder repair."""

from __future__ import annotations

from typing import Any

from .repair_engine import RepairEngine as _BaseRepairEngine
from .repairability import source_repair_block_reason


class RepairEngine(_BaseRepairEngine):
    """Repair engine variant that never asks the coder to patch non-source failures."""

    def _request_patch(
        self,
        evidence: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        build = evidence.get("build")
        diagnostics = evidence.get("diagnostics")
        reason = source_repair_block_reason(
            build=build if isinstance(build, dict) else None,
            diagnostics=diagnostics if isinstance(diagnostics, dict) else None,
        )
        if reason is not None:
            print(
                f"  [!] Source repair suppressed for non-source failure: {reason}",
                flush=True,
            )
            return []
        return super()._request_patch(evidence, context)
