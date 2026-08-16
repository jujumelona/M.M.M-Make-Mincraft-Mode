from __future__ import annotations

from functools import wraps
from typing import Any


def install(pagination_module: Any) -> None:
    """Safe no-op to allow planner_pagination_safety_contract to auto-suffix duplicate batch IDs without crashing."""
    pass


__all__ = ["install"]
