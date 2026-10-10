from __future__ import annotations

"""Executable regression routing for the master requirements ledger.

This module owns *how* a REG row is executed. It deliberately does not own
acceptance truth: a collected/passing structural regression is not a substitute
for model, Fabric, benchmark, or repeated-E2E receipts required by an ACC gate.
"""

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


@dataclass(frozen=True)
class RegressionExecutionRoute:
    pytest_target: str
    evidence_scope: str = "regression"


EXPECTED_REGRESSION_IDS = frozenset(
    [*(f"REG-{index:03d}" for index in range(1, 38)), "REG-SEM-001", "REG-DESIGN-001"]
)


REGRESSION_EXECUTION_ROUTES: dict[str, RegressionExecutionRoute] = {
    "REG-001": RegressionExecutionRoute("tests/test_root_cause_trace_durable.py"),
    "REG-002": RegressionExecutionRoute("tests/test_plan_collect_all_linker.py"),
    "REG-003": RegressionExecutionRoute("tests/test_plan_collect_all_linker.py"),
    "REG-004": RegressionExecutionRoute(
        "tests/test_mutation_authority.py::test_exact_authority_allows_only_host_paths"
    ),
    "REG-005": RegressionExecutionRoute("tests/test_agent_capability_manifest_degradation.py"),
    "REG-006": RegressionExecutionRoute("tests/test_task_artifact_contract.py"),
    "REG-007": RegressionExecutionRoute(
        "tests/test_missing_creation_slot_regression.py::"
        "test_fresh_host_reserved_task_cannot_silently_fall_back_without_binding"
    ),
    "REG-008": RegressionExecutionRoute("tests/test_plan_collect_all_linker.py"),
    "REG-009": RegressionExecutionRoute("tests/test_root_cause_trace_durable.py"),
    "REG-010": RegressionExecutionRoute("tests/test_llama_completion_liveness_contract.py::test_second_semantic_progress_timeout_is_not_retried_forever"),
    "REG-011": RegressionExecutionRoute("tests/test_progress_loop_liveness_regression.py"),
    "REG-012": RegressionExecutionRoute(
        "tests/test_source_set_boundary_contract.py::test_common_source_cannot_import_project_client_source"
    ),
    "REG-013": RegressionExecutionRoute("tests/test_resource_host_version_rules.py::test_missing_host_leaf_binding_fails_closed"),
    "REG-014": RegressionExecutionRoute("tests/test_managed_content_gametest.py::test_runtime_receipt_rejects_mutated_test_source_and_recipe"),
    "REG-015": RegressionExecutionRoute("tests/test_runtime_json_gap_regression.py"),
    "REG-016": RegressionExecutionRoute("tests/test_structured_output_backend_json_recovery.py"),
    "REG-017": RegressionExecutionRoute(
        "tests/test_model_runtime_performance_contract.py",
        evidence_scope="structural_harness",
    ),
    "REG-018": RegressionExecutionRoute("tests/test_requirement_query_rewrite_contract.py"),
    "REG-019": RegressionExecutionRoute("tests/test_rag_index_reuse_efficiency.py"),
    "REG-020": RegressionExecutionRoute("tests/test_resource_asset_backend_contract.py"),
    "REG-021": RegressionExecutionRoute("tests/test_semantic_prerequisite_edge_regression.py::test_space_progression_prerequisite_graph_cannot_collapse_to_zero_edges"),
    "REG-022": RegressionExecutionRoute(
        "tests/test_evidence_first_session_integration.py::test_one_requirement_can_bind_every_semantic_slice_without_fixed_ref_cap"
    ),
    "REG-023": RegressionExecutionRoute(
        "tests/test_agent_context_window_contract.py",
        evidence_scope="structural_harness",
    ),
    "REG-024": RegressionExecutionRoute("tests/test_qwen_parser_single_owner.py"),
    "REG-025": RegressionExecutionRoute("tests/test_semantic_source_partition_regression.py::test_empty_semantic_graph_cannot_claim_full_source_coverage"),
    "REG-026": RegressionExecutionRoute("tests/test_target_snapshot_hardening.py"),
    "REG-027": RegressionExecutionRoute("tests/test_verifier_receipt_truth_contract.py"),
    "REG-028": RegressionExecutionRoute("tests/test_platform_lock_naming_regime.py"),
    "REG-029": RegressionExecutionRoute("tests/test_target_semantics_boundary.py"),
    "REG-030": RegressionExecutionRoute(
        "tests/test_resolved_version_context.py::test_reuse_rejects_missing_or_different_context"
    ),
    "REG-031": RegressionExecutionRoute("tests/test_verifier_receipt_truth_contract.py"),
    "REG-032": RegressionExecutionRoute("tests/test_project_inventory_contract.py"),
    "REG-033": RegressionExecutionRoute(
        "tests/test_missing_creation_slot_regression.py::"
        "test_fresh_production_binding_cannot_create_a_source_outside_owned_paths"
    ),
    "REG-034": RegressionExecutionRoute(
        "tests/test_packaged_jar_required_content_regression.py::test_packaged_jar_missing_required_runtime_resource_fails"
    ),
    "REG-035": RegressionExecutionRoute("tests/test_ci_quality_regression_audit.py::test_new_function_hard_ceiling_is_rejected"),
    "REG-036": RegressionExecutionRoute("tests/test_agent_security_contract.py"),
    "REG-037": RegressionExecutionRoute(
        "tests/test_work_receipt_integrity.py::test_corrupt_task_receipt_is_invalidated_with_descendants_on_resume"
    ),
    "REG-SEM-001": RegressionExecutionRoute(
        "tests/test_semantic_prerequisite_edge_regression.py::test_missing_space_progression_prerequisite_is_rejected_not_dropped"
    ),
    "REG-DESIGN-001": RegressionExecutionRoute(
        "tests/test_evidence_first_session_integration.py::test_one_requirement_can_bind_every_semantic_slice_without_fixed_ref_cap"
    ),
}


def pytest_targets(regression_ids: Iterable[str] | None = None) -> tuple[str, ...]:
    selected = (
        sorted(REGRESSION_EXECUTION_ROUTES)
        if regression_ids is None
        else tuple(regression_ids)
    )
    return tuple(REGRESSION_EXECUTION_ROUTES[item].pytest_target for item in selected)


def _parse_target(target: str) -> tuple[str, str | None]:
    parts = target.split("::")
    if len(parts) > 2:
        raise ValueError("pytest target may contain at most one function selector")
    relative = parts[0]
    selector = parts[1] if len(parts) == 2 else None
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "tests"
        or path.suffix != ".py"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("pytest target must be a canonical tests/*.py path")
    if selector is not None and (
        not selector.startswith("test_")
        or not selector.isidentifier()
    ):
        raise ValueError("pytest selector must name one top-level test_* function")
    return relative, selector


def audit_regression_execution_routes(
    repository_root: str | Path,
    *,
    expected_ids: Iterable[str] = EXPECTED_REGRESSION_IDS,
) -> tuple[str, ...]:
    """Statically prove every registered pytest target resolves to executable tests."""

    root = Path(repository_root).resolve(strict=True)
    expected = frozenset(expected_ids)
    actual = frozenset(REGRESSION_EXECUTION_ROUTES)
    issues: list[str] = []

    for missing in sorted(expected - actual):
        issues.append(f"missing executable regression route: {missing}")
    for extra in sorted(actual - expected):
        issues.append(f"unexpected executable regression route: {extra}")

    parsed_files: dict[str, frozenset[str]] = {}
    for regression_id, route in sorted(REGRESSION_EXECUTION_ROUTES.items()):
        try:
            relative, selector = _parse_target(route.pytest_target)
        except ValueError as exc:
            issues.append(f"{regression_id}: {exc}")
            continue
        path = root / relative
        try:
            path.relative_to(root)
            source = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
            issues.append(f"{regression_id}: pytest target unavailable: {relative}: {type(exc).__name__}")
            continue
        if relative not in parsed_files:
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError as exc:
                issues.append(f"{regression_id}: pytest target is not valid Python: {relative}: {exc.msg}")
                continue
            parsed_files[relative] = frozenset(
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            )
        tests = parsed_files.get(relative, frozenset())
        if not tests:
            issues.append(f"{regression_id}: pytest target contains no top-level tests: {relative}")
        elif selector is not None and selector not in tests:
            issues.append(f"{regression_id}: pytest selector does not exist: {route.pytest_target}")

    return tuple(issues)


__all__ = [
    "EXPECTED_REGRESSION_IDS",
    "REGRESSION_EXECUTION_ROUTES",
    "RegressionExecutionRoute",
    "audit_regression_execution_routes",
    "pytest_targets",
]
