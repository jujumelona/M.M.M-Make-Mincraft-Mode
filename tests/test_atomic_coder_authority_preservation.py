from __future__ import annotations

import json

from minecraft_mod_ai import small_model_atomic_coder_execution as atomic
from minecraft_mod_ai.implementation_template_contract import SCHEMA as CODER_EXECUTION_SCHEMA


def _messages() -> list[dict[str, str]]:
    target = "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    anchor = {
        "kind": "symbol",
        "locator": target + "#DebugToken",
        "status": "host_reserved",
        "ownership": "task",
        "module_id": "debug_token",
        "source_set": "main",
    }
    contract = {
        "schema_version": CODER_EXECUTION_SCHEMA,
        "task_ref": "debug_token",
        "task_sha256_input": "sha256:task",
        "contract_sha256": "sha256:contract",
        "semantic_outcome": "add one deterministic debug token item",
        "execution_role": "coder",
        "requirement_refs": ["req-debug-token"],
        "target_constraints": {"exact_paths_only": True},
        "targets": [{"locator": anchor["locator"], "kind": "symbol"}],
        "depends_on": [],
        "dataflow": {"consumes": [], "provides": ["debug_token"]},
        "engineering_worksheet": {"api": "fabric"},
        "artifacts": [{"path": target}],
        "reuse_refs": [],
        "protected_boundaries": {"writable_paths": [target]},
        "verification_plan": [{"gate": "target_compile"}],
        "completion_predicate": {"kind": "all_gates_pass"},
        "implementation_steps": [
            {
                "sequence": 1,
                "obligation": "create debug token source",
                "target_refs": [anchor["locator"]],
                "consumes": [],
                "must_provide": ["debug_token"],
                "execution_checklist": ["create exact owned target"],
                "done_when": "target compiles",
            }
        ],
    }
    request = {
        "phase": "implement_module",
        "workspace_project_root": ".",
        "module": {
            "module_id": "debug_token",
            "kind": "custom_java",
            "evidence_task": {
                "task_id": "debug_token",
                "task_sha256": "sha256:task",
                "owned_anchors": [anchor],
                "production_bindings": [
                    {
                        "task_ref": "debug_token",
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
                "required_gates": ["target_compile"],
                "coder_execution_contract": contract,
            },
        },
    }
    return [
        {"role": "system", "content": "coder"},
        {"role": "user", "content": json.dumps(request)},
    ]


def test_atomic_lowering_preserves_host_task_authority_exactly() -> None:
    batches = atomic.atomicize_coder_messages(_messages())
    assert len(batches) == 1
    payload = json.loads(batches[0][1]["content"])
    task = payload["module"]["evidence_task"]

    assert task["task_id"] == "debug_token"
    assert task["task_sha256"] == "sha256:task"
    assert task["required_gates"] == ["target_compile"]
    assert task["owned_anchors"] == [
        {
            "kind": "symbol",
            "locator": "src/main/java/dev/mmm/debugfixture/DebugToken.java#DebugToken",
            "status": "host_reserved",
            "ownership": "task",
            "module_id": "debug_token",
            "source_set": "main",
        }
    ]
    assert task["production_bindings"][0]["task_ref"] == "debug_token"
    assert task["production_bindings"][0]["reuse_action"] == "fresh"
    assert task["production_bindings"][0]["owned_anchors"] == task["owned_anchors"]


def test_atomic_lowering_never_substitutes_observation_path_for_owned_target() -> None:
    messages = _messages()
    payload = json.loads(messages[1]["content"])
    payload["initial_exact_source_context"] = {
        "files": [
            {
                "path": "src/main/java/dev/mmm/debugfixture/MMMDebugFixture.java",
                "content": "class MMMDebugFixture {}",
            }
        ]
    }
    messages[1]["content"] = json.dumps(payload)

    batch = atomic.atomicize_coder_messages(messages)[0]
    lowered = json.loads(batch[1]["content"])
    task = lowered["module"]["evidence_task"]
    owned = task["owned_anchors"][0]["locator"]

    assert owned.startswith("src/main/java/dev/mmm/debugfixture/DebugToken.java#")
    assert "MMMDebugFixture.java" not in owned
