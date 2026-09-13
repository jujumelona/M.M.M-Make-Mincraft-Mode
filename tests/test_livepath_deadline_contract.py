from __future__ import annotations

import inspect
import math

from minecraft_mod_ai import artifact_graph_executor
from minecraft_mod_ai import model_router
from minecraft_mod_ai import parallel_model_tasks
from minecraft_mod_ai import planning_state_implementation
from minecraft_mod_ai import pre_design_rag_corrective
from minecraft_mod_ai import pre_design_research_pipeline


def test_parallel_model_map_uses_shared_deadline_executor() -> None:
    source = inspect.getsource(parallel_model_tasks.deterministic_model_map)
    assert "iter_completed_with_deadlines" in source
    assert "ThreadPoolExecutor" not in source
    assert "enumerate(values)" in source


def test_detailed_planning_wait_is_deadline_bounded() -> None:
    source = inspect.getsource(planning_state_implementation._compile_requirement_plans_dag)
    assert "DETAILED_PLAN_SECTION_TIMEOUT" in source
    assert "planning_work_unit_timeout_seconds" in source
    assert "timeout=timeout" in source
    assert "future.result(timeout=0)" in source
    assert "shutdown(wait=False, cancel_futures=True)" in source
    assert "with ThreadPoolExecutor" not in source


def test_artifact_generation_and_validation_are_deadline_bounded() -> None:
    source = inspect.getsource(artifact_graph_executor.execute_artifact_graph)
    assert "ARTIFACT_GRAPH_WORK_TIMEOUT" in source
    assert "future_deadlines" in source
    assert "timeout=timeout" in source
    assert "future.result(timeout=0)" in source
    assert "generation_pool.shutdown(wait=False, cancel_futures=True)" in source
    assert "validation_pool.shutdown(wait=False, cancel_futures=True)" in source
    assert "with ThreadPoolExecutor" not in source


def test_artifact_single_job_fast_path_keeps_model_deadline() -> None:
    source = inspect.getsource(artifact_graph_executor.execute_artifact_graph)
    branch = source.index("if len(ordered_jobs) == 1")
    parallel = source.index("order = {job.job_id: index", branch)
    single_job_source = source[branch:parallel]
    assert "planning_work_unit_timeout_seconds()" in single_job_source
    assert "run_with_model_execution_deadline" in single_job_source
    assert "_validate_completed_job" in single_job_source


def test_agent_tool_round_limit_is_not_a_default_completion_rule(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    assert math.isinf(model_router._agent_tool_round_limit())
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "7")
    assert model_router._agent_tool_round_limit() == 7


def test_corrective_rag_default_completion_is_semantic(monkeypatch) -> None:
    monkeypatch.delenv("MMM_PREDESIGN_CORRECTIVE_ROUNDS", raising=False)
    assert pre_design_rag_corrective._corrective_round_limit() is None
    source = inspect.getsource(pre_design_rag_corrective._quality_research_document_domain)
    assert "while True" in source
    assert "corrective_round_limit_reached" not in source
    assert "PREDESIGN_CORRECTIVE_SAFETY_LIMIT" in source


def test_predesign_domains_use_deadline_aware_model_map() -> None:
    source = inspect.getsource(pre_design_research_pipeline.collect_design_research)
    assert "deterministic_model_map" in source
    assert "thread_name_prefix=\"predesign-domain\"" in source
