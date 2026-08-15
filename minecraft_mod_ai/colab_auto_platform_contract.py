from __future__ import annotations

"""Deprecated compatibility import.

Public sessions no longer inject a historical 1.20.1/Fabric constructor placeholder,
so there is nothing for a Colab-specific wrapper to remove.
"""

from typing import Any


def install(game_design_module: Any) -> None:
    del game_design_module


__all__ = ["install"]
