from __future__ import annotations

from .model_router import ModelRouter
from .planning_pipeline import PlanningPipeline
from .spec import Proposal


class RoutedPlanner:
    """Planner protocol adapter backed by the canonical host planning compiler."""

    def __init__(self, *, profile: str = "t4_local", router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter(profile=profile)
        self.last_game_design: dict[str, object] | None = None

    def plan(self, prompt: str) -> Proposal:
        artifacts = PlanningPipeline(self.router).prepare(prompt)
        self.last_game_design = dict(artifacts.game_design)
        return artifacts.base_proposal
