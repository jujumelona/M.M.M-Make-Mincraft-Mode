from __future__ import annotations

import inspect

from minecraft_mod_ai.agent_roles import skills_for_model_role
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.host_grounding import build_coder_grounding


def _grounding(kind: str = "networking") -> dict:
    return build_coder_grounding(
        module_kind=kind,
        source_observation_receipt={
            "schema_version": "mmm/source-observation-receipt-v1",
            "project_sha256": "sha256:" + "1" * 64,
            "query_sha256": "sha256:" + "2" * 64,
            "observation_count": 3,
            "observations_sha256": "sha256:" + "3" * 64,
        },
        research_context={
            "schema_version": "mmm/module-research-context-v1",
            "corpus_sha256": "sha256:" + "4" * 64,
            "ledger_fact_count": 8,
            "selected_record_count": 2,
            "selected_fact_count": 2,
            "selected_facts_sha256": "sha256:" + "5" * 64,
            "omitted_fact_count": 6,
        },
        minecraft_version="1.20.1",
        loader="fabric",
        mappings="1.20.1+build.1",
    )


def test_host_grounding_is_mandatory_and_role_scoped() -> None:
    grounding = _grounding()
    policy = grounding["policy"]
    assert policy["resolved_before_first_coder_decode"] is True
    assert policy["baseline_grounding_owned_by_host"] is True
    assert policy["baseline_grounding_optional_for_model"] is False
    assert policy["model_tool_choice_required_for_baseline"] is False
    assert policy["retrieved_context_can_authorize"] is False
    assert "ground-production-with-live-evidence" in grounding["required_skills"]
    assert "patch-existing-project" in grounding["required_skills"]
    assert "generate-gui-networking" in grounding["required_skills"]
    assert set(grounding["required_skills"]).issubset(set(skills_for_model_role("coder")))
    assert grounding["reviewed_mcp_servers"]
    assert grounding["evidence_bindings"]["project_exact_rag"]["request_field"] == "relevant_context"
    assert grounding["evidence_bindings"]["approved_research_rag"]["request_field"] == "research_context"


def test_generator_resolves_grounding_before_agentic_coder_decode() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    grounding_index = source.index("host_grounding = build_coder_grounding(")
    first_decode_index = source.index("self.router.generate_text(")
    assert grounding_index < first_decode_index
    assert '"host_grounding": host_grounding' in source
    assert 'response_format="text"' in source
    assert 'tool_stage="generation"' in source
    assert "enable_tools=True" in source
    assert "generate_tool_decision" not in source
    assert '"phase": "implement_module"' in source
    assert "plan_files" not in source
    budget_guard = source.index("project_context_budget = _coder_project_context_budget(")
    exact_observation = source.index("_collect_initial_observations(")
    assert budget_guard < exact_observation < grounding_index
    assert "fast_path_express" not in source
