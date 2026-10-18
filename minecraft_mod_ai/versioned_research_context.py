from __future__ import annotations

"""Target-selected project research context without external mod repositories.

Only code in the current project is eligible for this evidence lane.
Minecraft/Fabric API authority is resolved separately from the selected target.
"""

from .research_code_context import Evidence, PlanStep, ResearchCodeContext


class VersionedResearchCodeContext(ResearchCodeContext):
    """Preserve the existing caller interface with project-local evidence only."""

    def _retrieve_repo_examples(
        self, query: str, *, plan_step: PlanStep | None
    ) -> list[Evidence]:
        local = super()._retrieve_repo_examples(query, plan_step=plan_step)
        for item in local:
            item.metrics["current_project_authority"] = 1.0
            item.bestfit_score = max(item.bestfit_score, 0.80)
        return local


__all__ = ["VersionedResearchCodeContext"]
