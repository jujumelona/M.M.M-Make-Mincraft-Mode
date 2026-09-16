from __future__ import annotations

import json

from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready
from minecraft_mod_ai.host_grounding import _SCHEMA_VERSION


def _messages(*, selected_fact_count: int):
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
    return [
        {
            "role": "user",
            "content": json.dumps({"host_grounding": grounding}),
        }
    ]


def test_empty_research_does_not_bypass_required_fresh_retrieval():
    assert host_baseline_evidence_ready(_messages(selected_fact_count=0)) is False


def test_selected_research_can_satisfy_host_baseline_grounding():
    assert host_baseline_evidence_ready(_messages(selected_fact_count=1)) is True
