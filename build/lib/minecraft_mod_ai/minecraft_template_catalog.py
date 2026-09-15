from __future__ import annotations

"""Public structural Minecraft artifact-routing API.

All routing logic lives in ``structural_routing_contract`` and accepts only typed
structural obligations. Semantic feature names, domains, reference titles, and gameplay
archetypes are intentionally not part of this API.
"""

from .structural_routing_contract import (
    ARTIFACT_DEPENDENCIES,
    CANONICAL_ARTIFACT_KINDS,
    FLAG_TO_ARTIFACT,
    StructuralArtifactPlan,
    StructuralArtifactRequirements,
    build_artifact_plan,
    detect_artifacts,
    expand_artifact_dependencies,
    requirements_from_record,
    validate_artifact_kinds,
)

__all__ = [
    "ARTIFACT_DEPENDENCIES",
    "CANONICAL_ARTIFACT_KINDS",
    "FLAG_TO_ARTIFACT",
    "StructuralArtifactPlan",
    "StructuralArtifactRequirements",
    "build_artifact_plan",
    "detect_artifacts",
    "expand_artifact_dependencies",
    "requirements_from_record",
    "validate_artifact_kinds",
]
