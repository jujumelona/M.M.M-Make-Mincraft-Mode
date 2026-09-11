import pytest

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionError,
    _semantic_execution_observations,
)
from minecraft_mod_ai.complete_spec import ProductionModule


def _module(module_id: str) -> ProductionModule:
    return ProductionModule(
        module_id=module_id,
        kind="item",
        config={
            "evidence_task": {
                "task_sha256": "sha256:" + module_id[0] * 64,
                "requirement_refs": [module_id],
                "gap_refs": [],
                "impact_probes": [],
            }
        },
    )


def _by_task(observations):
    return {
        item["task_id"]: (item["patch_receipt"], tuple(item["touched_paths"]))
        for item in observations
    }


def test_grouped_receipt_uses_explicit_module_ownership_not_position():
    members = [_module("alpha"), _module("beta"), _module("gamma")]
    grouped = {
        "module_ids": ["beta", "alpha"],
        "operation_count": 2,
        "touched_paths": ["src/grouped.java"],
        "patch_receipt": "grouped",
    }
    solo = {
        "module_id": "gamma",
        "operation_count": 1,
        "touched_paths": ["src/gamma.java"],
        "patch_receipt": "solo",
    }
    expected = {
        "alpha": ("grouped", ("src/grouped.java",)),
        "beta": ("grouped", ("src/grouped.java",)),
        "gamma": ("solo", ("src/gamma.java",)),
    }
    forward = _semantic_execution_observations(
        members,
        [grouped, solo],
        downstream_ids=lambda _module_id: (),
    )
    reverse = _semantic_execution_observations(
        members,
        [solo, grouped],
        downstream_ids=lambda _module_id: (),
    )
    assert _by_task(forward) == expected
    assert _by_task(reverse) == expected


def test_unowned_receipt_fails_closed_in_multi_module_node():
    members = [_module("alpha"), _module("beta")]
    with pytest.raises(
        CompleteProductionError,
        match="SEMANTIC_RECEIPT_OWNERSHIP_MISSING",
    ):
        _semantic_execution_observations(
            members,
            [
                {
                    "patch_receipt": "ambiguous",
                    "touched_paths": ["src/unknown.java"],
                }
            ],
            downstream_ids=lambda _module_id: (),
        )


def test_receipt_declaring_foreign_owner_fails_closed():
    members = [_module("alpha"), _module("beta")]
    with pytest.raises(
        CompleteProductionError,
        match="SEMANTIC_RECEIPT_OWNER_OUTSIDE_NODE.*gamma",
    ):
        _semantic_execution_observations(
            members,
            [{"module_id": "gamma", "patch_receipt": "foreign"}],
            downstream_ids=lambda _module_id: (),
        )


def test_tracked_module_without_any_owned_receipt_fails_closed():
    members = [_module("alpha"), _module("beta")]
    with pytest.raises(
        CompleteProductionError,
        match="SEMANTIC_RECEIPT_COVERAGE_MISSING.*beta",
    ):
        _semantic_execution_observations(
            members,
            [{"module_id": "alpha", "patch_receipt": "alpha-only"}],
            downstream_ids=lambda _module_id: (),
        )
