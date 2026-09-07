from __future__ import annotations

"""Compatibility import surface for the canonical coder execution contract.

The complete coder hand-off is owned by ``implementation_template_contract``.  This
module intentionally contains no second payload builder, hash implementation, defaults,
or schema literal.  Older imports may continue to use this name, but every caller receives
the exact same canonical contract object shape.
"""

from collections.abc import Mapping
from typing import Any

from .implementation_template_contract import (
    SCHEMA as CODER_EXECUTION_CONTRACT_SCHEMA,
    _validate_contract,
    build_implementation_template,
)

build_coder_execution_contract = build_implementation_template


def project_task_for_coder(task: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap one canonical coder contract in the minimal evidence-task envelope."""

    contract = build_coder_execution_contract(task)
    result: dict[str, Any] = {
        "task_id": contract["task_ref"],
        "coder_execution_contract": contract,
    }
    source_sha = str(task.get("task_sha256") or "").strip()
    if source_sha:
        result["task_sha256"] = source_sha
    return result


__all__ = [
    "CODER_EXECUTION_CONTRACT_SCHEMA",
    "_validate_contract",
    "build_coder_execution_contract",
    "project_task_for_coder",
]
