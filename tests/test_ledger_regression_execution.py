from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.ledger_regression_execution import (
    EXPECTED_REGRESSION_IDS,
    REGRESSION_EXECUTION_ROUTES,
    audit_regression_execution_routes,
)
from minecraft_mod_ai.ledger_trace_registry import (
    REGRESSION_MANIFEST,
    validate_executable_manifest_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_all_ledger_regressions_have_executable_routes() -> None:
    validate_executable_manifest_snapshot()
    assert set(REGRESSION_MANIFEST) == EXPECTED_REGRESSION_IDS
    assert set(REGRESSION_EXECUTION_ROUTES) == EXPECTED_REGRESSION_IDS
    assert len(REGRESSION_EXECUTION_ROUTES) == 39
    assert all(
        route.execution_status == "executable"
        for route in REGRESSION_MANIFEST.values()
    )
    assert all(
        REGRESSION_MANIFEST[regression_id].test_case == route.pytest_target
        for regression_id, route in REGRESSION_EXECUTION_ROUTES.items()
    )
    assert audit_regression_execution_routes(REPOSITORY_ROOT) == ()


def test_source_set_regression_is_bound_to_direct_common_client_failure() -> None:
    assert REGRESSION_EXECUTION_ROUTES["REG-012"].pytest_target == (
        "tests/test_source_set_boundary_contract.py::"
        "test_common_source_cannot_import_project_client_source"
    )


def test_packaged_jar_regression_is_bound_to_missing_runtime_resource_failure() -> None:
    assert REGRESSION_EXECUTION_ROUTES["REG-034"].pytest_target == (
        "tests/test_packaged_jar_required_content_regression.py::"
        "test_packaged_jar_missing_required_runtime_resource_fails"
    )


def test_structural_model_harnesses_do_not_claim_runtime_acceptance() -> None:
    assert REGRESSION_EXECUTION_ROUTES["REG-017"].evidence_scope == "structural_harness"
    assert REGRESSION_EXECUTION_ROUTES["REG-023"].evidence_scope == "structural_harness"
