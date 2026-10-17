from __future__ import annotations
"""CurseForge-only catalogue handoff; repository donor projection removed."""
from collections.abc import Mapping, Sequence
from typing import Any

def project_repository_candidates(domain: Mapping[str, Any], grounded: Mapping[str, Any], *, per_domain_limit: int | None = None) -> list[dict[str, Any]]:
    """Catalogue results are descriptive evidence, not source repositories."""
    return []

def merge_repository_candidates(existing: Sequence[Mapping[str, Any]] | None, new: Sequence[Mapping[str, Any]] | None, *, global_limit: int | None = None) -> list[dict[str, Any]]:
    """No repository identities are promoted from catalogue metadata."""
    return []

def authored_design_repository_candidates(content_design: Mapping[str, Any]) -> list[dict[str, Any]]:
    return []

def planning_state_repository_cards(design: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return ()

def ensure_authored_source_handoff(content_design: Mapping[str, Any], *, router: Any, prompt: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Preserve catalogue research receipts without repository donor selection."""
    from .pre_design_research_pipeline import _canonical_sha256, collect_source_candidate_handoff
    content = dict(content_design)
    expected = _canonical_sha256(prompt)
    old = content.get("_pre_design_research")
    if isinstance(old, Mapping) and old.get("prompt_sha256") != expected:
        content.pop("_pre_design_research", None)
        state = content.get("_planning_state")
        if isinstance(state, Mapping):
            content["_planning_state"] = {k: v for k, v in state.items() if k != "repository_candidates"}
    cached = content.get("_pre_design_source_handoff")
    if not isinstance(cached, Mapping) or cached.get("prompt_sha256") != expected:
        try:
            cached = collect_source_candidate_handoff(router, prompt)
        except Exception as exc:
            cached = {"schema_version":"mmm/pre-design-source-candidate-handoff-v1", "prompt_sha256":expected, "repository_candidates":[], "status":"provider_error", "error_type":type(exc).__name__}
    content["_pre_design_source_handoff"] = dict(cached)
    return content, []

__all__ = ["project_repository_candidates", "merge_repository_candidates", "authored_design_repository_candidates", "planning_state_repository_cards", "ensure_authored_source_handoff"]
