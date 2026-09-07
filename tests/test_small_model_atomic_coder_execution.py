from __future__ import annotations

import json

from minecraft_mod_ai.small_model_atomic_coder_execution import atomicize_coder_messages


def _messages(*, step_count: int = 3):
    steps = [
        {
            "sequence": index,
            "obligation": f"Implement isolated behavior {index + 1}.",
            "target_refs": ["src/main/java/demo/Feature.java#Feature"],
            "execution_checklist": ["x"] * 8,
        }
        for index in range(step_count)
    ]
    request = {
        "phase": "implement_module",
        "task": "Implement the entire task.",
        "module": {
            "module_id": "task_feature",
            "kind": "custom_java",
            "evidence_task": {
                "task_id": "task_feature",
                "task_sha256": "sha256:" + "a" * 64,
                "engineering_worksheet": {"huge": "never expose to atomic coder" * 200},
                "research_reuse_candidates": ["host-only" * 200],
                "coder_execution_contract": {
                    "schema_version": "mmm/coder-execution-contract-v2",
                    "task_ref": "task_feature",
                    "task_sha256_input": "sha256:" + "a" * 64,
                    "semantic_outcome": "Feature behaves exactly as approved.",
                    "target_constraints": {
                        "minecraft_version": "1.21.1",
                        "loader": "fabric",
                        "mappings": "yarn",
                        "java_version": "21",
                    },
                    "depends_on": ["task_core"],
                    "dataflow": {
                        "consumes": ["core_ready"],
                        "provides": ["feature_ready"],
                    },
                    "targets": [
                        {
                            "path": "src/main/java/demo/Feature.java",
                            "locator": "src/main/java/demo/Feature.java#Feature",
                        }
                    ],
                    "implementation_steps": steps,
                    "engineering_worksheet": {"huge": "also never expose" * 200},
                    "verification_plan": [{"gate": "target_compile"}],
                },
            },
        },
        "initial_exact_source_context": {
            "observations": [{"path": "Feature.java", "content": "class Feature {}"}]
        },
        "research_context": {"selected_facts": ["verified API fact"]},
        "host_grounding": {"schema_version": "mmm/host-owned-coder-grounding-v1"},
        "rules": ["Use tools."],
    }
    return (
        {"role": "system", "content": "coder"},
        {"role": "user", "content": json.dumps(request)},
    )


def _request(batch):
    return json.loads(batch[-1]["content"])


def test_multi_obligation_task_becomes_one_model_batch_per_obligation() -> None:
    batches = atomicize_coder_messages(_messages(step_count=3))

    assert len(batches) == 3
    obligations = [
        _request(batch)["module"]["evidence_task"]["coder_execution_contract"]["step"]["obligation"]
        for batch in batches
    ]
    assert obligations == [
        "Implement isolated behavior 1.",
        "Implement isolated behavior 2.",
        "Implement isolated behavior 3.",
    ]
    assert all(
        _request(batch)["atomic_execution"]["policy"]
        == "one_model_call_one_implementation_obligation"
        for batch in batches
    )


def test_atomic_batch_does_not_leak_sibling_obligations_or_host_blobs() -> None:
    batches = atomicize_coder_messages(_messages(step_count=3))

    first = batches[0][-1]["content"]
    assert "Implement isolated behavior 1." in first
    assert "Implement isolated behavior 2." not in first
    assert "Implement isolated behavior 3." not in first
    assert "engineering_worksheet" not in first
    assert "research_reuse_candidates" not in first
    assert "never expose" not in first
    assert "verification_plan" not in first


def test_later_atomic_steps_do_not_receive_stale_pre_step_source_page() -> None:
    batches = atomicize_coder_messages(_messages(step_count=3))

    first = _request(batches[0])
    second = _request(batches[1])
    third = _request(batches[2])
    assert first["initial_exact_source_context"]["observations"]
    for request in (second, third):
        assert request["initial_exact_source_context"] == {
            "mode": "retrieve_current_atomic_step_with_tools",
            "reason": (
                "Earlier atomic steps may have changed the staged workspace; do not replay "
                "the stale pre-step source page."
            ),
        }


def test_single_step_is_still_compacted_to_atomic_contract() -> None:
    batches = atomicize_coder_messages(_messages(step_count=1))

    assert len(batches) == 1
    request = _request(batches[0])
    contract = request["module"]["evidence_task"]["coder_execution_contract"]
    assert contract["schema_version"] == "mmm/atomic-coder-step-v1"
    assert contract["step"] == {
        "index": 1,
        "count": 1,
        "obligation": "Implement isolated behavior 1.",
    }
    assert "implementation_steps" not in contract


def test_unrelated_model_request_is_not_rewritten() -> None:
    messages = ({"role": "user", "content": "plain question"},)
    assert atomicize_coder_messages(messages) == (messages,)
