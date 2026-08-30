from __future__ import annotations

"""Single fail-closed source-license policy for reusable donor code.

Only canonical SPDX identifiers explicitly admitted here may advance from repository
discovery into reusable-source proof. Unknown identifiers and compound expressions
are rejected until an explicit policy is implemented for them.
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
    """Return the canonical candidate spelling used by the strict allowlist."""

    return str(value or "").strip()


def is_reusable_source_license(value: Any) -> bool:
    """Return True only for an explicitly admitted canonical source license."""

    return normalize_source_license(value) in REUSABLE_SOURCE_LICENSES


__all__ = [
    "REUSABLE_SOURCE_LICENSES",
    "is_reusable_source_license",
    "normalize_source_license",
]
