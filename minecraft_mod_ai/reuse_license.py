from __future__ import annotations

"""Fail-closed access to the source-transplant donor-license authority.

Source discovery owns the canonical permissive SPDX allowlist. Proof code reaches
that same policy through this narrow predicate so a directly constructed DonorSlice
cannot bypass the discovery-time license gate.
"""

from typing import Any


def normalize_source_license(value: Any) -> str:
    """Return the canonical candidate spelling used by the strict allowlist."""

    return str(value or "").strip()


def is_reusable_source_license(value: Any) -> bool:
    """Return True only for a license admitted by source-transplant discovery."""

    # Import lazily so proof-level definitions do not pull the discovery stack into
    # module import time. source_transplant is the existing discovery authority.
    from .source_transplant import _PERMISSIVE

    return normalize_source_license(value) in _PERMISSIVE


__all__ = [
    "is_reusable_source_license",
    "normalize_source_license",
]
