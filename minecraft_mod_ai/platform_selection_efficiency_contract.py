from __future__ import annotations

"""Deprecated compatibility import.

Platform selection is owned by platform_resolver/platform_optimizer.  This module is
kept only until runtime_bootstrap import compatibility is removed; it installs no
wrapper and cannot alter target choice.
"""

from typing import Any


def install(*, resolver_module: Any, central_contract_module: Any) -> None:
    del resolver_module, central_contract_module


__all__ = ["install"]
