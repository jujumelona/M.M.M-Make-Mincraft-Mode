from __future__ import annotations

"""Reconcile deterministic generation boundaries after runtime composition.

Game design is host-owned and deterministic.  Runtime finalization therefore installs
only host-side generation contracts: the approval-bound Fabric platform lock and the
resource-asset preflight that must run before any prompt/image model work.
"""

import json
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False


def _write_approval_bound_bootstrap_lock(
    root: Path,
    adapter: Any,
    receipt: dict[str, Any],
) -> None:
    """Use the canonical immutable writer, then attach bootstrap evidence."""
    from . import platform_generation_contract

    platform_generation_contract._write_platform_lock(root, adapter)
    target = Path(root) / ".minecraft_ai" / "platform-lock.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Canonical platform lock writer did not produce an object.")
    payload["bootstrap"] = dict(receipt)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import fabric_official_template_provider as fabric_provider
    from . import resource_asset_production
    from .resource_asset_preflight_contract import install as install_resource_asset_preflight

    install_resource_asset_preflight(resource_asset_production)

    original_platform_lock_writer = fabric_provider._write_platform_lock
    if not getattr(original_platform_lock_writer, "_mmm_approval_bound_bootstrap_lock", False):

        @wraps(original_platform_lock_writer)
        def write_platform_lock(root: Path, adapter: Any, receipt: dict[str, Any]) -> None:
            _write_approval_bound_bootstrap_lock(root, adapter, receipt)

        write_platform_lock._mmm_approval_bound_bootstrap_lock = True
        write_platform_lock.__wrapped__ = original_platform_lock_writer
        fabric_provider._write_platform_lock = write_platform_lock

    _INSTALLED = True


__all__ = ["install"]
