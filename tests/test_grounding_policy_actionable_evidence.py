from __future__ import annotations

import json

from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready
from minecraft_mod_ai.host_grounding import _SCHEMA_VERSION


def _messages(*, selected_fact_count: int, fresh_java: bool):
    grounding = {
        "schema_version": _SCHEMA_VERSION,
        "policy": {
            "resolved_before_first_coder_decode": True,
            "baseline_grounding_owned_by_host": True,
            "baseline_grounding_optional_for_model": False,
            "model_tool_choice_required_for_baseline": False,
        },
        "evidence_bindings": {
            "project_exact_rag": {
                "receipt": {
                    "project_sha256": "project-sha",
                    "observations_sha256": "observations-sha",
                }
            },
            "approved_research_rag": {
                "receipt": {
                    "selected_fact_count": selected_fact_count,
                }
            },
        },
    }
    messages = []
    if fresh_java:
        messages.append(
            {
                "role": "developer",
                "content": json.dumps(
                    {
                        "schema_version": "mmm/small-model-task-capsule",
                        "reuse_action": "fresh",
                        "mutation_target": {
                            "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java"
                        },
                    }
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": json.dumps({"host_grounding": grounding}),
        }
    )
    return messages


def test_fresh_java_with_empty_research_requires_retrieval():
    assert (
        host_baseline_evidence_ready(
            _messages(selected_fact_count=0, fresh_java=True)
        )
        is False
    )


def test_fresh_java_with_selected_research_can_use_host_grounding():
    assert (
        host_baseline_evidence_ready(
            _messages(selected_fact_count=1, fresh_java=True)
        )
        is True
    )


def test_nonfresh_turn_preserves_existing_host_grounding_contract():
    assert (
        host_baseline_evidence_ready(
            _messages(selected_fact_count=0, fresh_java=False)
        )
        is True
    )
