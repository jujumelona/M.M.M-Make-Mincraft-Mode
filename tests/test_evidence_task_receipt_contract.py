from __future__ import annotations

from dataclasses import dataclass

import pytest

import minecraft_mod_ai.evidence_task_receipt_contract as contract
from minecraft_mod_ai.resource_asset_production import _validate_evidence_module_binding


@dataclass(frozen=True)
class _Module:
    config: dict
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()


def _semantic_plan() -> dict:
    return {
        "plan_sha256": "plan-sha",
        "request_catalog": {
            "prompt_sha256": "prompt-sha",
            "requirements": [
                {
                    "requirement_id": "req_a",
                    "capability": "cap_a",
                    "statement": "Implement A",
                }
            ],
        },
        "tasks": [
            {
                "task_id": "task_a",
                "requirement_refs": ["req_a"],
                "gap_refs": ["gap_a"],
                "reuse_refs": [],
                "owned_anchors": [{"kind": "test", "locator": "Test.java#Test"}],
                "consumes": ["target:frozen"],
                "provides": ["capability:cap_a"],
                "acceptance": ["observable A"],
                "impact_probes": ["changed_symbols"],
                "depends_on": [],
                "required_gates": ["target_compile"],
            }
        ],
    }


def _execution_task() -> dict:
    task = dict(_semantic_plan()["tasks"][0])
    task.update(
        {
            "execution_role": "runtime_behavior",
            "derived_requirements": [{"requirement_id": "derived_a"}],
            "implementation_obligations": ["Implement exact runtime behavior A"],
            "owned_anchors": [
                {
                    "kind": "symbol",
                    "locator": "src/main/java/example/TaskA.java#TaskA",
                },
                {"kind": "test", "locator": "Test.java#Test"},
            ],
        }
    )
    return task


def _canonical_handoff() -> dict:
    return {
        "source_plan_sha256": "plan-sha",
        "handoff_sha256": "handoff-sha",
        "work_graph": {"task_refs": ["task_a"], "edges": []},
        "production_modules": [],
        "asset_requests": [],
    }


def _execution_handoff() -> dict:
    return {
        **_canonical_handoff(),
        "execution_overlay_sha256": "execution-overlay-sha",
        "production_modules": [
            {
                "production_module_id": "pm-a",
                "task_ref": "task_a",
                "module_id": "root",
                "source_set": "main",
            }
        ],
        "asset_requests": [],
    }


def _patch_lowering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "validate_evidence_first_plan", lambda _plan: None)
    monkeypatch.setattr(
        contract,
        "validate_evidence_first_handoff",
        lambda _handoff, source_plan=None: None,
    )
    monkeypatch.setattr(
        contract,
        "execution_plan",
        lambda _plan: {**_plan, "tasks": [_execution_task()]},
    )
    monkeypatch.setattr(
        contract,
        "execution_handoff",
        lambda _plan, _canonical, _lowered: _execution_handoff(),
    )
    monkeypatch.setattr(contract, "validate_plan_collect_all", lambda *_args: None)


def test_execution_receipt_bundle_owns_producer_and_validator_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lowering(monkeypatch)
    bundle = contract.build_execution_receipt_bundle(
        _semantic_plan(), canonical_handoff=_canonical_handoff()
    )
    receipt = bundle["receipts"]["task_a"]

    assert receipt["execution_role"] == "runtime_behavior"
    assert receipt["derived_requirements"] == [{"requirement_id": "derived_a"}]
    assert receipt["implementation_obligations"] == ["Implement exact runtime behavior A"]
    assert receipt["execution_overlay_sha256"] == "execution-overlay-sha"
    assert receipt["production_bindings"] == _execution_handoff()["production_modules"]
    assert receipt["request_context"] == {
        "prompt_sha256": "prompt-sha",
        "requirements": _semantic_plan()["request_catalog"]["requirements"],
        "derived_requirements": [{"requirement_id": "derived_a"}],
    }
    contract.validate_task_receipt(receipt, expected_receipt=receipt)


def test_execution_receipt_validator_fails_closed_on_schema_or_value_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lowering(monkeypatch)
    expected = contract.build_execution_receipt_bundle(
        _semantic_plan(), canonical_handoff=_canonical_handoff()
    )["receipts"]["task_a"]

    stale = {**expected, "execution_overlay_sha256": "stale"}
    with pytest.raises(ValueError, match="execution_overlay_sha256"):
        contract.validate_task_receipt(stale, expected_receipt=expected)

    unknown = {**expected, "future_unowned_field": True}
    with pytest.raises(ValueError, match="unrecognized receipt fields"):
        contract.validate_task_receipt(unknown, expected_receipt=expected)

    missing = dict(expected)
    missing.pop("implementation_obligations")
    with pytest.raises(ValueError, match="missing receipt fields"):
        contract.validate_task_receipt(missing, expected_receipt=expected)


def test_resource_binding_accepts_exact_execution_receipt_instead_of_semantic_task() -> None:
    semantic_task = _semantic_plan()["tasks"][0]
    execution_task = _execution_task()
    expected_receipt = {
        **execution_task,
        "handoff_sha256": "handoff-sha",
        "execution_overlay_sha256": "execution-overlay-sha",
        "production_bindings": [],
        "asset_bindings": [],
        "request_context": {
            "prompt_sha256": "prompt-sha",
            "requirements": [],
            "derived_requirements": execution_task["derived_requirements"],
        },
    }
    config = {
        "evidence_plan_sha256": "plan-sha",
        "evidence_task": expected_receipt,
        "batch_id": "task_a",
        **{
            key: execution_task[key]
            for key in (
                "requirement_refs",
                "gap_refs",
                "reuse_refs",
                "owned_anchors",
                "consumes",
                "provides",
                "acceptance",
                "impact_probes",
            )
        },
    }
    module = _Module(
        config=config,
        depends_on=tuple(execution_task["depends_on"]),
        required_gates=tuple(execution_task["required_gates"]),
    )

    # No requirement decision is needed for this isolated binding regression.  The exact
    # log failure happened before reuse ownership logic; an empty semantic ref set isolates
    # the producer/consumer schema boundary under test.
    semantic_without_refs = {**semantic_task, "requirement_refs": [], "reuse_refs": []}
    execution_without_refs = {
        **expected_receipt,
        "requirement_refs": [],
        "reuse_refs": [],
    }
    module.config["evidence_task"] = execution_without_refs
    module.config["requirement_refs"] = []
    module.config["reuse_refs"] = []

    _validate_evidence_module_binding(
        module=module,
        semantic_task=semantic_without_refs,
        expected_receipt=execution_without_refs,
        evidence_plan_sha256="plan-sha",
        requirements={},
        decisions={},
        components={},
    )
