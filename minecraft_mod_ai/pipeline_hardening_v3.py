from __future__ import annotations

"""Compatibility repair for machine-only Minecraft pack metadata discovery."""

from typing import Any

from .pipeline_hardening import _replace_bound_references

_INSTALLED = False


def install_pipeline_hardening_v3() -> None:
    """Preserve the official pack resolver's three-value ABI without article fallbacks."""

    global _INSTALLED
    if _INSTALLED:
        return

    from . import platform_live_discovery as live

    current = live._official_pack_versions
    if not getattr(current, "_mmm_machine_pack_contract_v3", False):
        def machine_only(version: str) -> tuple[str, str, str]:
            target = str(version or "").strip()
            data_pack, resource_pack = live._mojang_pack_versions(target)
            return data_pack, resource_pack, live._mojang_target_url(target)

        machine_only._mmm_machine_pack_contract_v3 = True  # type: ignore[attr-defined]
        live._official_pack_versions = machine_only
        _replace_bound_references(current, machine_only)

    _INSTALLED = True


__all__ = ["install_pipeline_hardening_v3"]
