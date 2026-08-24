from __future__ import annotations

"""Retired compatibility hook for the former second-stage tool selector guard.

Mandatory evidence and writable capabilities are now preserved directly inside
``small_model_max_agent_contract.select_tool_schemas`` so selection has one owner.
"""

from typing import Any


def install(max_agent_owner: Any) -> None:
    del max_agent_owner


__all__ = ["install"]
