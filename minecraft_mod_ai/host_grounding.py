from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping

from .agent_capability_context import reviewed_mcp_servers_for_model_role
from .agent_roles import skills_for_model_role

_SCHEMA_VERSION = "mmm/host-owned-coder-grounding-v1"
_KIND_SKILL: dict[str, str] = {
    "recipe": "generate-datagen",
    "advancement": "generate-datagen",
    "loot": "generate-datagen",
    "structure": "generate-worldgen",
    "biome": "generate-worldgen",
    "dimension": "generate-worldgen",
    "world_event": "generate-worldgen",
    "entity": "generate-geckolib-entity",
    "boss": "generate-geckolib-entity",
    "npc": "generate-geckolib-entity",
    "quest": "generate-quest-progression",
    "class": "generate-quest-progression",
    "skill": "generate-quest-progression",
    "economy": "generate-quest-progression",
    "shop": "generate-quest-progression",
    "party": "generate-quest-progression",
    "guild": "generate-quest-progression",
    "gui": "generate-gui-networking",
    "networking": "generate-gui-networking",
}
_ALLOWED_WRITE_PREFIXES = (
    "src/main/java/",
    "src/main/resources/",
    "src/test/java/",
    "src/gametest/",
    ".minecraft_ai/",
)
_ALLOWED_WRITE_FILES = ("build.gradle", "gradle.properties", "settings.gradle")
_PROTECTED_WRITE_PREFIXES = (
    ".minecraft_ai/research",
    ".minecraft_ai/context-observations",
)
_REJECTED_WRITE_EXAMPLES = ("README.md", "LICENSE", "docs/")


def custom_module_write_scope() -> dict[str, Any]:
    """Return the single model/validator write-scope contract."""

    return {
        "allowed_prefixes": list(_ALLOWED_WRITE_PREFIXES),
        "allowed_files": list(_ALLOWED_WRITE_FILES),
        "protected_prefixes": list(_PROTECTED_WRITE_PREFIXES),
        "examples_rejected": list(_REJECTED_WRITE_EXAMPLES),
        "policy": "Every patch operation path must match this allowlist.",
    }


def _normalized_scope_path(path: str) -> str:
    raw = str(path).replace("\\", "/").strip()
    if not raw:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    normalized = candidate.as_posix()
    return "" if normalized in {"", "."} else normalized


def custom_module_path_protected(path: str) -> bool:
    normalized = _normalized_scope_path(path).casefold()
    if not normalized:
        return False
    return any(
        normalized == root or normalized.startswith(root + "/")
        for root in _PROTECTED_WRITE_PREFIXES
    )


def custom_module_path_allowed(path: str) -> bool:
    normalized = _normalized_scope_path(path)
    if not normalized or custom_module_path_protected(normalized):
        return False
    return normalized in _ALLOWED_WRITE_FILES or any(
        normalized.startswith(prefix) for prefix in _ALLOWED_WRITE_PREFIXES
    )


def build_coder_grounding(
    *,
    module_kind: str,
    source_observation_receipt: Mapping[str, Any],
    research_context: Mapping[str, Any],
    minecraft_version: str,
    loader: str,
    mappings: str,
) -> dict[str, Any]:
    """Build the compact code-owned grounding contract for a coder turn.

    This function deliberately does not perform network retrieval. The durable
    research stage owns external evidence collection, while ProjectIndex owns exact
    project-source retrieval. Generation receives both products before the first
    coder decode together with the reviewed Skill/MCP execution routes. This keeps
    baseline grounding mandatory without duplicating expensive project scans or
    external requests for every production shard.

    The exact write allowlist is also published here before the first decode. Keeping
    it beside the evidence bindings makes the model-facing contract match the host
    validator and avoids expensive repair generations for paths such as ``README.md``
    that the patcher can never accept.
    """
    kind = str(module_kind).strip()
    if not kind:
        raise ValueError("module_kind must be non-empty")

    eligible_skills = tuple(skills_for_model_role("coder"))
    eligible_set = set(eligible_skills)
    specialized = _KIND_SKILL.get(kind, "generate-fabric-core")
    required = _ordered_existing(
        (
            "ground-production-with-live-evidence",
            "patch-existing-project",
            specialized,
        ),
        eligible_set,
    )
    reviewed_servers = tuple(
        sorted(reviewed_mcp_servers_for_model_role("generation", "coder"))
    )
    source_receipt = {
        key: source_observation_receipt.get(key)
        for key in (
            "schema_version",
            "project_sha256",
            "query_sha256",
            "observation_count",
            "observations_sha256",
        )
        if key in source_observation_receipt
    }
    research_receipt = {
        key: research_context.get(key)
        for key in (
            "schema_version",
            "corpus_sha256",
            "ledger_fact_count",
            "selected_record_count",
            "selected_fact_count",
            "selected_facts_sha256",
            "omitted_fact_count",
        )
        if key in research_context
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "stage": "generation",
        "model_role": "coder",
        "target": {
            "minecraft_version": str(minecraft_version),
            "loader": str(loader),
            "mappings": str(mappings),
            "java": "17",
        },
        "required_skills": list(required),
        "reviewed_mcp_servers": list(reviewed_servers),
        "evidence_bindings": {
            "project_exact_rag": {
                "request_field": "relevant_context",
                "receipt": source_receipt,
            },
            "approved_research_rag": {
                "request_field": "research_context",
                "receipt": research_receipt,
            },
        },
        "write_scope": custom_module_write_scope(),
        "policy": {
            "resolved_before_first_coder_decode": True,
            "baseline_grounding_owned_by_host": True,
            "baseline_grounding_optional_for_model": False,
            "model_tool_choice_required_for_baseline": False,
            "retrieved_context_can_authorize": False,
            "writes_still_require_approved_pipeline": True,
            "supplemental_retrieval_after_host_validation": True,
        },
    }


def _ordered_existing(names: tuple[str, ...], eligible: set[str]) -> tuple[str, ...]:
    result: list[str] = []
    for name in names:
        if name in eligible and name not in result:
            result.append(name)
    if "ground-production-with-live-evidence" not in result:
        raise RuntimeError(
            "MinecraftCoder is missing ground-production-with-live-evidence Skill."
        )
    return tuple(result)


__all__ = [
    "build_coder_grounding",
    "custom_module_path_allowed",
    "custom_module_path_protected",
    "custom_module_write_scope",
]
