from __future__ import annotations

"""Platform-neutral planner prompts and selected-evidence handoff.

Target selection has exactly one owner:
``platform_central_ai_contract`` -> ``platform_resolver`` -> ``platform_optimizer``.
This contract does not select coordinates and does not perform a second target RAG
when the selection owner already produced evidence.
"""

from functools import wraps
from typing import Any, Mapping


_PROMPT_REPLACEMENTS = (
    ("Minecraft Java 1.20.1 Fabric", "the host-selected Minecraft Java target"),
    ("Minecraft 1.20.1 Fabric", "the host-selected Minecraft target"),
    (
        "the host-selected Minecraft Java Fabric target",
        "the host-selected Minecraft Java target",
    ),
    ("the host-selected Minecraft Fabric target", "the host-selected Minecraft target"),
)


def install(
    *,
    game_design_module: Any,
    complete_planner_module: Any,
    central_research_module: Any,
) -> None:
    _install_target_neutral_game_design_prompts(game_design_module)
    _neutralize_complete_planner_prompts(complete_planner_module)
    _install_selected_target_evidence(
        complete_planner_module,
        central_research_module,
    )


def _install_target_neutral_game_design_prompts(module: Any) -> None:
    original_system = module._system_prompt
    if not getattr(original_system, "_mmm_target_neutral_prompt", False):

        @wraps(original_system)
        def system_prompt() -> str:
            text = original_system().replace(
                "GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.",
                "GameDesignPlanner for a Minecraft Java mod production system.",
            )
            return text + (
                "\n\nPlatform rule: describe capabilities only. Do not choose or assume a "
                "Minecraft version, loader, mappings, Java, build tool, or package coordinate. "
                "The host resolves and verifies the target after semantic design."
            )

        system_prompt._mmm_target_neutral_prompt = True
        module._system_prompt = system_prompt

    original_sharded = module._sharded_design_system_prompt
    if not getattr(original_sharded, "_mmm_target_neutral_prompt", False):

        @wraps(original_sharded)
        def sharded_prompt() -> str:
            text = original_sharded().replace(
                "request for a Minecraft Java 1.20.1 Fabric mod.",
                "request for a Minecraft Java mod.",
            )
            return text + (
                "\nReturn semantic requirements only; exact platform coordinates are host-owned."
            )

        sharded_prompt._mmm_target_neutral_prompt = True
        module._sharded_design_system_prompt = sharded_prompt


def _neutralize_complete_planner_prompts(module: Any) -> None:
    """Fold the old platform_prompt_contract behavior into the one planning contract."""

    for name, value in tuple(vars(module).items()):
        if not name.startswith("_") or not isinstance(value, str):
            continue
        updated = value
        for old, new in _PROMPT_REPLACEMENTS:
            updated = updated.replace(old, new)
        if updated != value:
            setattr(module, name, updated)


def _install_selected_target_evidence(module: Any, central: Any) -> None:
    def retrieve_implementation_evidence(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = game_design.get("_platform_evidence")
        if isinstance(existing, Mapping):
            return dict(existing)

        # Deserialized/legacy plans may predate the evidence handoff. Only those
        # paths perform one target-scoped fallback retrieval here.
        brief = research_brief or central.normalize_research_brief(prompt, game_design)
        selection = game_design.get("_platform_selection")
        target = selection.get("target") if isinstance(selection, Mapping) else None
        if not isinstance(target, Mapping):
            raise module.SpecValidationError("Planning target selection is missing.")
        if not isinstance(brief.get("_mmm_platform_target"), Mapping):
            brief = {**brief, "_mmm_platform_target": dict(target)}
        return central.retrieve_domain_evidence(brief)

    retrieve_implementation_evidence._mmm_selected_target_evidence = True
    module._retrieve_implementation_evidence = retrieve_implementation_evidence


__all__ = ["install"]
