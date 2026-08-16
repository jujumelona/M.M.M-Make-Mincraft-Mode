from __future__ import annotations

import re
from typing import Any


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def install(planner_module: Any) -> None:
    """Safe no-op to allow production module normalizers to work without crashing on ID formatting."""
    pass


__all__ = ["install"]
