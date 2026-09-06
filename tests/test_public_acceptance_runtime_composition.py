from __future__ import annotations

from dataclasses import dataclass
from inspect import unwrap
from typing import Any

from minecraft_mod_ai import complete_planner, production_contract
from minecraft_mod_ai import evidence_first_pipeline_contract as pipeline_contract


@dataclass(frozen=True)
class _Batch:
    batch_id: str
    scope: str
    depends_on_batches: tuple[str, ...]
    deliverables: tuple[str, ...]
    exports: tuple[str, ...]
    task_contract: dict[str, Any] | None = None
    evidence_plan_sha256: str = ""
    acceptance_tests: tuple[str, ...] = ()


def test_execution_receipt_keeps_task_acceptance_internal(monkeypatch) -> None:
    task_id = "task_alien_planet_interaction_semantic_im_47278ef7e7"
    internal_acceptance = (
        f"{task_id}: all declared provides exist and all owned anchors pass their integrity checks"
    )
    plan = {"plan_sha256": "sha256:" + "1" * 64}
    receipt = {
        "task_id": task_id,
        "semantic_outcome": "Implement alien planet interaction",
        "depends_on": [],
        "provides": ["capability:alien_planet_interaction"],
        "requirement_refs": ["req_alien_planet_interaction"],
        "acceptance": [internal_acceptance],
    }
    monkeypatch.setattr(
        pipeline_contract,
        "build_execution_receipt_bundle",
        lambda _value: {
            "plan_sha256": plan["plan_sha256"],
            "task_refs": (task_id,),
            "dependencies": {task_id: ()},
            "receipts": {task_id: receipt},
        },
    )

    batches = pipeline_contract._batches_from_handoff(plan, batch_type=_Batch)

    assert len(batches) == 1
    assert batches[0].acceptance_tests == ()
    assert batches[0].task_contract is not None
    assert batches[0].task_contract["acceptance"] == [internal_acceptance]


def test_complete_planner_resolves_finalized_production_contract_dynamically() -> None:
    canonical = unwrap(complete_planner.CompleteGameDesignPlanner._plan_in_session)
    globals_map = canonical.__globals__

    assert globals_map["production_contract"] is production_contract
    assert "compile_production_contract" not in globals_map
