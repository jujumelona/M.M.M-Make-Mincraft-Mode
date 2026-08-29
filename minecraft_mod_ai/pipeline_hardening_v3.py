from __future__ import annotations

"""Compatibility marker for verified Minecraft pack metadata discovery."""

_INSTALLED = False


def install_pipeline_hardening_v3() -> None:
    """Preserve the current official resolver instead of replacing its ABI.

    ``platform_live_discovery._official_pack_versions`` already uses Mojang's
    machine-readable, checksummed ``version.json`` as the primary source and keeps
    bounded official release metadata as a fail-closed fallback. Replacing that
    function here used to strip its ``lru_cache`` control API and duplicated stale
    behavior. Keep the canonical resolver intact and only mark the installed policy.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from . import platform_live_discovery as live

    current = live._official_pack_versions
    setattr(current, "_mmm_machine_pack_contract_v3", True)
    _INSTALLED = True


__all__ = ["install_pipeline_hardening_v3"]
