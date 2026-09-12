from __future__ import annotations

import inspect

from minecraft_mod_ai import artifact_graph_executor
from minecraft_mod_ai import parallel_model_tasks
from minecraft_mod_ai import planning_state_implementation


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


def test_artifact_single_job_has_no_unbounded_fast_path() -> None:
    source = inspect.getsource(artifact_graph_executor.execute_artifact_graph)
    assert "if len(ordered_jobs) == 1" not in source
