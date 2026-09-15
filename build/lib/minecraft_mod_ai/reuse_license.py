from __future__ import annotations

"""Canonical fail-closed license policy for reusable external source donors.

Both discovery and executable proof must consult this module. Keeping the SPDX
allowlist here avoids circular authority, duplicated policy, and silent drift between
candidate admission and proof-time verification.
"""

from typing import Any

REUSABLE_SOURCE_LICENSES = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "Zlib",
        "Unlicense",
        "CC0-1.0",
    }
)


def normalize_source_license(value: Any) -> str:
    """Return the exact SPDX-style spelling used by the strict allowlist."""

    return str(value or "").strip()


def is_reusable_source_license(value: Any) -> bool:
    """Return True only for licenses explicitly admitted for source reuse."""

    return normalize_source_license(value) in REUSABLE_SOURCE_LICENSES


__all__ = [
    "REUSABLE_SOURCE_LICENSES",
    "is_reusable_source_license",
    "normalize_source_license",
]
