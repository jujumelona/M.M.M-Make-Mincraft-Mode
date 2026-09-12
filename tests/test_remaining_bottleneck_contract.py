from __future__ import annotations

import inspect

from minecraft_mod_ai import (
    planning_state_adaptive_implementation,
    resource_asset_production,
    source_transplant,
)


def test_source_transplant_deadline_workers_bind_donor_identity() -> None:
    source = inspect.getsource(source_transplant.materialize_source_slices)
    assert "iter_completed_with_deadlines" in source
    assert "executor.map" not in source
    assert "shutdown(wait=True" not in source
    assert "repository_name" in source
    assert "pinned_commit" in source


def test_asset_generation_stops_after_first_selectable_candidate() -> None:
    source = inspect.getsource(resource_asset_production.generate_assets)
    assert "selected = None" in source
    assert "attempted_candidate_count" in source
    assert "scored.append((1000.0" not in source
    selected = source.index("selected = (1000.0, index, normalized, evidence)")
    assert source.index("break", selected) > selected


def test_planning_artifact_receipt_cannot_claim_runtime_pass() -> None:
    normalize = planning_state_adaptive_implementation._planning_only_artifact_receipt
    receipt = normalize({"status": "PASS", "proof": {"passed": True, "scope": "static"}})
    assert receipt["status"] == "PLANNED"
    assert receipt["verification_status"] == "PENDING_RUNTIME_VALIDATION"
    assert "proof" not in receipt

    normalized = normalize({"status": "PLANNED", "planned_validation": {"scope": "static"}})
    assert normalized["status"] == "PLANNED"
    assert normalized["verification_status"] == "PENDING_RUNTIME_VALIDATION"
