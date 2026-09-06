"""Stable import surface for canonical MMM contract types.

Do not define duplicate hand-off shapes here. Contract ownership stays in the
existing subsystem contract modules; this package exposes a predictable path
for agents and tooling to discover those canonical definitions.
"""

from .target import TargetContract

__all__ = ["TargetContract"]
