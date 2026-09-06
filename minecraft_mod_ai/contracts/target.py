"""Canonical target contract import surface.

This module intentionally re-exports the existing TargetContract so all new
code has one stable contract path without duplicating the type definition.
"""

from ..target_contract import TargetContract

__all__ = ["TargetContract"]
