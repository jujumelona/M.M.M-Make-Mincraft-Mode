from __future__ import annotations

from .pipeline import MinecraftModPipeline
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_validator import ScalableProjectValidator
from .scale_policy import ScalePolicy


class ScalableMinecraftModPipeline(MinecraftModPipeline):
    """Compatibility pipeline with the approved legacy flow and scalable internals.

    Approval binding, broker authorization, Gradle, GameTest, JAR validation and release
    packaging remain inherited. Only the old monolithic generator and hard-coded source
    budgets are replaced by deterministic shards and explicit host resource policy.
    """

    def __init__(self, *, planner=None, broker=None, policy: ScalePolicy | None = None) -> None:
        super().__init__(planner=planner, broker=broker)
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.generator = ScalableFabricProjectGenerator(policy=self.policy)
        self.validator = ScalableProjectValidator(policy=self.policy)
