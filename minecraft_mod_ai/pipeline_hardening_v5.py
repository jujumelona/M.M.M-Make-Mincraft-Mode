from __future__ import annotations

"""Compatibility no-op for the retired pre-design checkpoint hardening stage."""

_INSTALLED = False


def install_pipeline_hardening_v5() -> None:
    global _INSTALLED
    _INSTALLED = True


__all__ = ["install_pipeline_hardening_v5"]
