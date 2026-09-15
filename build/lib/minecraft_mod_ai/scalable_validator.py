from __future__ import annotations

from pathlib import Path

from .scale_policy import ScalePolicy
from .validator import ProjectValidator, ValidationReport


class ScalableProjectValidator:
    """Policy-native source validator kept as the complete-pipeline public name."""

    def __init__(self, *, policy: ScalePolicy | None = None) -> None:
        self.policy = policy or ScalePolicy.from_environment()
        self.validator = ProjectValidator(policy=self.policy)

    def validate(self, root: Path, spec) -> ValidationReport:
        return self.validator.validate(root, spec)
