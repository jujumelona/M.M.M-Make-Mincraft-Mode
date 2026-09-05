from __future__ import annotations

"""Static planning-depth helpers for the host-owned template compiler.

Task-DAG construction and cross-system dependency binding are owned directly by
``evidence_first_planning``.  This compatibility facade therefore exposes historical
helper imports without installing a second planner or monkey-patching ``_compile_tasks``.
"""

from .. import game_design
from ..planning_depth_helpers import (
    _compile_pre_retrieval_plan_with_design_facets,
    _design_facets,
    _facet_work_index,
    _semantic_model_with_leaf_decomposition,
)

_INSTALLED = False


def _production_depth_game_design_prompt() -> str:
    return (
        game_design._system_prompt()
        + "\n\nPRODUCTION-DEPTH DESIGN CONTRACT:\n"
        "- Complete the gameplay/mod design before choosing any third-party implementation "
        "or ecosystem donor. Search and reuse happen only after this design is frozen.\n"
        "- Preserve authored scope, but expand requested mechanics into the smallest meaningful "
        "subsystems that can be independently implemented, tested, and searched for reuse.\n"
        "- Use as many module entries as the design needs; there is no arbitrary module count.\n"
        "- core_loop and progression must expose prerequisites, state changes, costs/rewards, "
        "and unlock transitions in executable order.\n"
        "- Acceptance tests must cover mechanics that can fail independently.\n"
        "- Record assumptions precisely enough for downstream reviewed Skills/MCP research "
        "to verify exact Minecraft/Fabric APIs, versions, licenses, and reusable implementations.\n"
        "- Do not add unrelated features merely to make the design longer."
    )


def install() -> None:
    """Mark the compatibility facade installed without mutating planner callables."""

    global _INSTALLED
    _INSTALLED = True


__all__ = [
    "_compile_pre_retrieval_plan_with_design_facets",
    "_design_facets",
    "_facet_work_index",
    "_production_depth_game_design_prompt",
    "_semantic_model_with_leaf_decomposition",
    "install",
]
