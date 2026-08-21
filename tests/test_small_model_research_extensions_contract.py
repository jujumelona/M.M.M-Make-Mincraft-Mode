from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.small_model_research_extensions_contract import (
    _compact_executor_skill,
    _memory_route,
    _model_family,
    _project_capability_payload,
)


def test_memory_route_scales_only_when_needed() -> None:
    assert _memory_route("find registry id", "general", 6) == ("exact", 3)
    assert _memory_route("fix compile error in registry", "repair", 6) == (
        "targeted",
        6,
    )
    deep_query = "\n".join(
        [
            "cross-file integration dependency chain regression",
            *[f"constraint_{index} symbol_{index}" for index in range(80)],
        ]
    )
    tier, limit = _memory_route(deep_query, "repair", 6)
    assert tier == "deep"
    assert limit >= 8


def test_instruction_projection_keeps_only_frontier_skills() -> None:
    payload = {
        "eligible_skills": [
            {
                "name": "rag",
                "model_tools": ["search_code_rag"],
                "validators": ["evidence"],
            },
            {
                "name": "runtime",
                "model_tools": ["run_playtest"],
                "validators": ["runtime"],
            },
            {
                "name": "host-only",
                "model_tools": [],
                "host_owned_tools": ["package_release"],
            },
        ]
    }
    projected = _project_capability_payload(
        payload,
        exposed_tools=frozenset({"search_code_rag"}),
    )
    assert [item["name"] for item in projected["eligible_skills"]] == ["rag"]
    receipt = projected["instruction_projection"]
    assert receipt["candidate_skill_count"] == 3
    assert receipt["selected_skill_count"] == 1
    assert receipt["policy"] == "prompt_projection_only_authorization_unchanged"


def test_model_family_uses_registry_declared_executor_policy() -> None:
    assert _model_family(
        SimpleNamespace(
            model_id="other/model",
            extra={"executor_skill_family": "qwen3.5"},
        )
    ) == "qwen3.5"
    assert _model_family(
        SimpleNamespace(
            model_id="other/model",
            extra={"executor_skill_family": "QWEN3.6"},
        )
    ) == "qwen3.6"
    assert _model_family(
        {"executor_skill_family": "qwen3.6", "path": "/models/anything.gguf"}
    ) == "qwen3.6"

    # Model identity must not be inferred from repository ids or filenames. Runtime
    # behavior is owned by registry-declared adapter policy.
    assert _model_family(
        SimpleNamespace(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")
    ) == "generic"
    assert _model_family(
        SimpleNamespace(extra={"executor_skill_family": "unknown"})
    ) == "generic"


def test_executor_skill_rendering_is_compact_without_mutating_canonical() -> None:
    canonical = {
        "schema_version": "mmm/temporary-skill-v3",
        "ephemeral": True,
        "task_class": "repair",
        "current_query_terms": [f"term{index}" for index in range(40)],
        "procedural_hierarchy": {
            "function": {"procedures": ["inspect > edit"]},
            "subtask": {"procedures": ["retrieve > inspect > edit"]},
            "workflow": {"procedures": ["retrieve > inspect > edit > verify"]},
        },
        "proven_patterns": ["repair:java"],
        "avoid_patterns": ["repeat invalid patch"],
        "verifier_hints": ["jdt_status"],
        "source_trajectory_ids": ["sha256:abc"],
        "source_verification_levels": {"sha256:abc": "L4"},
        "rule": "verify current evidence",
    }

    rendered = _compact_executor_skill(canonical, "qwen3.5")
    assert canonical["source_trajectory_ids"] == ["sha256:abc"]
    assert "source_trajectory_ids" not in rendered
    assert len(rendered["current_query_terms"]) == 18
    assert "workflow" not in rendered["procedural_hierarchy"]
    assert rendered["executor_rendering"]["family"] == "qwen3.5"

    generic = _compact_executor_skill(canonical, "generic")
    assert generic["source_trajectory_ids"] == ["sha256:abc"]
