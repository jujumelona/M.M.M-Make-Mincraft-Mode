from __future__ import annotations

import inspect

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import _task_local_module_contract
from minecraft_mod_ai import runtime_bootstrap


def test_evidence_task_projection_does_not_replay_unrelated_global_config() -> None:
    task = {
        "task_id": "task_example",
        "semantic_outcome": "Implement one independently verifiable outcome",
        "requirement_refs": ["req_example"],
        "owned_anchors": [{"kind": "symbol", "locator": "src/main/java/X.java#X"}],
        "reuse_refs": [],
        "consumes": ["root:example"],
        "provides": ["example"],
        "depends_on": [],
        "acceptance": ["task_example: all declared provides exist"],
        "required_gates": ["source"],
        "impact_probes": ["changed_symbols"],
        "task_sha256": "sha256:" + "0" * 64,
    }
    module = ProductionModule(
        module_id="task_example",
        kind="custom_java",
        config={
            "implementation": "custom",
            "evidence_task": task,
            "unrelated_global_design": {"huge": "must stay host-side"},
        },
    )
    projected = _task_local_module_contract(module)
    assert projected["evidence_task"] == task
    assert "config" not in projected
    assert "unrelated_global_design" not in repr(projected)


def test_runtime_bootstrap_does_not_install_heuristic_tool_top_k() -> None:
    source = inspect.getsource(runtime_bootstrap._install_post_bootstrap_contracts)
    assert "_install_tool_retrieval" not in source
    assert "_install_repair_context" in source
