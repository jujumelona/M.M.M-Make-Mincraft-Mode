from __future__ import annotations

"""Platform-neutral planner prompts and selected-evidence handoff.

Exact coordinate selection is always delegated to ``platform_resolver`` ->
``platform_optimizer``. This contract also binds the legacy ``MinecraftModPipeline``
planning boundary to that same host-owned resolver so target-neutral semantic planners
are never validated or approved before a complete executable provider receipt exists.
"""

from contextvars import ContextVar
from functools import wraps
from typing import Any, Mapping


_SEMANTIC_PLANNING = ContextVar("mmm_semantic_platform_planning", default=False)

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
    _install_semantic_target_validation(game_design_module)
    _install_target_neutral_game_design_prompts(game_design_module)
    _neutralize_complete_planner_prompts(complete_planner_module)
    _install_selected_target_evidence(
        complete_planner_module,
        central_research_module,
    )
    _install_pipeline_target_binding()


def _install_semantic_target_validation(module: Any) -> None:
    """Permit an all-empty platform sentinel only inside semantic design.

    ``GameDesignPlanner`` validates its proposal before the central platform owner can
    attach a provider receipt.  The validation bypass is therefore context-local and
    only skips the unresolved platform leaf.  Every approval, generation, publishing,
    and ordinary ``Proposal.validate`` call keeps the strict fail-closed contract.
    """

    from .spec import PlatformLock

    current_lock_validate = PlatformLock.validate
    if not getattr(current_lock_validate, "_mmm_semantic_planning", False):

        @wraps(current_lock_validate)
        def validate_platform(lock: PlatformLock) -> None:
            if _SEMANTIC_PLANNING.get() and lock.is_unresolved():
                return
            current_lock_validate(lock)

        validate_platform._mmm_semantic_planning = True
        PlatformLock.validate = validate_platform

    cls = module.GameDesignPlanner
    current_plan = cls.plan
    if getattr(current_plan, "_mmm_semantic_planning", False):
        return

    @wraps(current_plan)
    def plan(self: Any, *args: Any, **kwargs: Any):
        token = _SEMANTIC_PLANNING.set(True)
        try:
            return current_plan(self, *args, **kwargs)
        finally:
            _SEMANTIC_PLANNING.reset(token)

    plan._mmm_semantic_planning = True
    cls.plan = plan


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


def _install_pipeline_target_binding() -> None:
    """Resolve a complete provider receipt before the legacy pipeline validates."""

    from . import pipeline as pipeline_module
    from . import platform_resolver

    cls = pipeline_module.MinecraftModPipeline
    current = cls.plan
    if getattr(current, "_mmm_host_target_binding", False):
        return

    @wraps(current)
    def plan(
        self: Any,
        prompt: str,
        *,
        existing_input: str | Any | None = None,
    ):
        proposal = self.planner.plan(prompt)
        report = None
        if existing_input is not None:
            report = pipeline_module.inspect_existing_project_archive(existing_input)

        selection = platform_resolver.resolve_platform(
            prompt,
            existing_version=(report.minecraft_version if report is not None else None),
            existing_loader=(report.loader if report is not None else None),
        )
        proposal = platform_resolver.retarget_proposal(proposal, selection)

        if report is not None:
            proposal = self._bind_existing_input(proposal, report)
        proposal.validate()
        return proposal

    plan._mmm_host_target_binding = True
    cls.plan = plan


__all__ = ["install"]
