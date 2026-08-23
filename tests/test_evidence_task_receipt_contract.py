from __future__ import annotations

from dataclasses import dataclass

import pytest

import minecraft_mod_ai.evidence_task_receipt_contract as contract


@dataclass(frozen=True)
class _Module:
    config: dict


def _plan() -> dict:
    return {
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
            }
        ],
    }


def _handoff() -> dict:
    return {
        "handoff_sha256": "handoff-sha",
        "work_graph": {"task_refs": ["task_a"], "edges": []},
        "production_modules": [
            {
                "production_module_id": "pm-a",
                "task_ref": "task_a",
                "module_id": "main",
                "source_set": "main",
            }
        ],
        "asset_requests": [
            {
                "asset_request_id": "asset-a",
                "task_ref": "task_a",
                "locator": "assets/example/a.png",
            }
        ],
    }


def test_build_task_receipt_extensions_uses_exact_handoff_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "validate_evidence_first_plan", lambda _plan: None)
    monkeypatch.setattr(
        contract,
        "validate_evidence_first_handoff",
        lambda _handoff, source_plan=None: None,
    )

    receipts = contract.build_task_receipt_extensions(_plan(), handoff=_handoff())

    assert receipts["task_a"] == {
        "handoff_sha256": "handoff-sha",
        "production_bindings": _handoff()["production_modules"],
        "asset_bindings": _handoff()["asset_requests"],
        "request_context": {
            "prompt_sha256": "prompt-sha",
            "requirements": _plan()["request_catalog"]["requirements"],
        },
    }


def test_validate_task_receipt_rejects_tampered_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "validate_evidence_first_plan", lambda _plan: None)
    monkeypatch.setattr(
        contract,
        "validate_evidence_first_handoff",
        lambda _handoff, source_plan=None: None,
    )
    task = _plan()["tasks"][0]
    expected = contract.build_task_receipt_extensions(_plan(), handoff=_handoff())["task_a"]
    embedded = {**task, **expected}

    contract.validate_task_receipt(
        embedded,
        task=task,
        expected_extensions=expected,
    )

    embedded["handoff_sha256"] = "stale"
    with pytest.raises(ValueError, match="handoff_sha256"):
        contract.validate_task_receipt(
            embedded,
            task=task,
            expected_extensions=expected,
        )


def test_legacy_validator_receives_compatibility_view_without_mutating_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minecraft_mod_ai import resource_asset_production as production

    plan = _plan()
    handoff = _handoff()
    expected = {
        "task_a": {
            "handoff_sha256": "handoff-sha",
            "production_bindings": handoff["production_modules"],
            "asset_bindings": handoff["asset_requests"],
            "request_context": {
                "prompt_sha256": "prompt-sha",
                "requirements": plan["request_catalog"]["requirements"],
            },
        }
    }
    task = plan["tasks"][0]
    original_receipt = {**task, **expected["task_a"]}
    module = _Module(config={"evidence_task": original_receipt})
    seen: list[dict] = []

    def legacy(*, module, task, **_kwargs):
        seen.append(module.config["evidence_task"])
        assert set(module.config["evidence_task"]) - set(task) == {"request_context"}

    monkeypatch.setattr(contract, "_INSTALLED", False)
    monkeypatch.setattr(contract, "build_task_receipt_extensions", lambda _plan: expected)
    monkeypatch.setattr(production, "_validate_evidence_module_binding", legacy)

    contract._install_reuse_receipt_guard()
    token = contract._ACTIVE_RECEIPTS.set((plan, expected))
    try:
        production._validate_evidence_module_binding(
            module=module,
            task=task,
            evidence_plan_sha256="plan-sha",
            request_catalog=plan["request_catalog"],
            requirements={"req_a": plan["request_catalog"]["requirements"][0]},
            decisions={},
            components={},
        )
    finally:
        contract._ACTIVE_RECEIPTS.reset(token)

    assert seen
    assert module.config["evidence_task"] == original_receipt
