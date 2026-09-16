from __future__ import annotations

"""Final owner for verifier-repair guidance after runtime composition.

Late runtime installers may replace methods on ``HostRunState``.  This module is applied
only after ``finalize_runtime()`` so the live repair guidance is deterministic and carries
the exact staged source that the verifier actually rejected.
"""

import hashlib
import json
from types import ModuleType
from typing import Any

_MARKER = "_mmm_verifier_repair_final_owner_v1"


def install(progress_module: ModuleType) -> None:
    if getattr(progress_module, _MARKER, False):
        return

    def take_verifier_repair_guidance(self: Any) -> str | None:
        with self._lock:
            if (
                self.validation_status != "FAIL"
                or not self.latest_verifier_fingerprint
                or self.latest_verifier_fingerprint == self.repair_guidance_fingerprint
            ):
                return None
            self.repair_guidance_fingerprint = self.latest_verifier_fingerprint
            context = self.mutation_context
            source = (
                context.source_body
                if context is not None and isinstance(context.source_body, str)
                else None
            )
            payload = {
                "verifier": self.latest_verifier_tool,
                "diagnostics": list(self.latest_verifier_errors),
                "target_path": context.target_path if context else None,
                "target_is_new_file": context.is_new_file if context else None,
                "writable_paths": list(context.writable_paths) if context else [],
                "current_source_sha256": (
                    hashlib.sha256(source.encode("utf-8")).hexdigest()
                    if source is not None
                    else None
                ),
                "current_source": source,
            }
        return (
            "MMM_CORE_VERIFIER_REPAIR_V5\n"
            "The verifier failure is the active repair obligation. Do not restart generation "
            "or search unrelated ecosystem candidates. The payload contains the exact host-tracked "
            "current source and its SHA-256. Repair only the host-pinned target path. For an existing "
            "Java target produced by the current mutation transaction, a same-path create_file payload "
            "with the complete corrected source may be materialized as a transactional replacement; "
            "it must never create or target a second path. Make one materially different repair, then "
            "return directly to VERIFY.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )

    take_verifier_repair_guidance.__name__ = "take_verifier_repair_guidance"
    progress_module.HostRunState.take_verifier_repair_guidance = take_verifier_repair_guidance
    setattr(progress_module, _MARKER, True)


def assert_installed(progress_module: ModuleType) -> None:
    if not getattr(progress_module, _MARKER, False):
        raise RuntimeError("final verifier repair owner is not installed")


__all__ = ["assert_installed", "install"]
