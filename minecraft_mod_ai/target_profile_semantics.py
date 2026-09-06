from __future__ import annotations

"""Deprecated import surface for target semantics.

Canonical ownership moved to :mod:`minecraft_mod_ai.target_contract`.  This module
contains no target policy; remaining callers are being migrated to the SSOT and this
surface can then be deleted without changing semantics.
"""

from .target_contract import (
    mappings_applicable,
    minecraft_version_tuple,
    minimum_java_major,
    uses_native_names,
)

__all__ = [
    "mappings_applicable",
    "minecraft_version_tuple",
    "minimum_java_major",
    "uses_native_names",
]
