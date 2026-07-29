from __future__ import annotations

from .game_design import GameDesignPlanner
from .model_router import ModelRouter
from .spec import Proposal


class RoutedPlanner:
    """Planner protocol adapter backed by the role registry; never falls back silently."""

    def __init__(self, *, profile: str = "t4_local", router: ModelRouter | None = None) -> None:
        self.router = router or ModelRouter(profile=profile)
        self.last_game_design: dict[str, object] | None = None

    def plan(self, prompt: str) -> Proposal:
        design, proposal = GameDesignPlanner(self.router).plan(prompt)
        self.last_game_design = design
        return proposal
