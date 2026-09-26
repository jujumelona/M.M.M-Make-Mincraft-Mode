from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.small_model_atomic_coder_execution import (
    AtomicCoderContractError,
    atomicize_coder_messages,
)

from minecraft_mod_ai.small_model_task_capsule_contract import (
    TaskAnchor,
    TaskCapsule,
    TaskCapsuleContractError,
    _atomic_request_scope,
    bind_source_edit_arguments,
)


def _messages(*, step_count: int = 3):
    checklist = [f"check-{index}" for index in range(8)]
    steps = [
        {
            "sequence": index,
            "obligation": f"Implement isolated behavior {index + 1}.",
            "target_refs": ["src/main/java/demo/Feature.java#Feature"],
            "consumes": ["core_ready"],
            "must_provide": ["feature_ready"],
            "execution_checklist": checklist,
            "done_when": "The isolated behavior is implemented and ready for host verification.",
        }
        for index in range(step_count)
    ]
    worksheet = {
        "runtime_behavior": {
            "status": "applicable",
            "requirements": ["Keep authoritative state server-side."],
        },
        "persistence": {
            "status": "applicable",
            "requirements": ["Preserve saved state across reload."],
        },
    }
    request = {
        "phase": "implement_module",
        "task": "Implement the entire task.",
        "module": {
            "module_id": "task_feature",
            "kind": "custom_java",
            "evidence_task": {
                "task_id": "task_feature",
                "task_sha256": "sha256:" + "a" * 64,
                "engineering_worksheet": {"outer_copy": "not authoritative"},
                "research_reuse_candidates": ["host-only" * 200],
                "coder_execution_contract": {
                    "schema_version": "mmm/coder-execution-contract",
                    "task_ref": "task_feature",
                    "task_sha256_input": "sha256:" + "a" * 64,
                    "contract_sha256": "sha256:" + "b" * 64,
                    "semantic_outcome": "Feature behaves exactly as approved.",
                    "execution_role": "production_with_verification",
                    "requirement_refs": ["REQ-feature"],
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
                        },
                        {
                            "path": "src/test/java/demo/FeatureTest.java",
                            "locator": "src/test/java/demo/FeatureTest.java#FeatureTest",
                        },
                    ],
                    "implementation_steps": steps,
                    "engineering_worksheet": worksheet,
                    "artifacts": [
                        {
                            "kind": "source_code",
                            "locator": "src/main/java/demo/Feature.java#Feature",
                        }
                    ],
                    "reuse_refs": ["reuse:selected"],
                    "protected_boundaries": {
                        "writable_paths": [
                            "src/main/java/demo/Feature.java",
                            "src/test/java/demo/FeatureTest.java",
                        ]
                    },
                    "verification_plan": [
                        {"gate": "target_compile", "executor": "host_gate_runner"},
                        {
                            "gate": "observable_acceptance",
                            "executor": "host_acceptance_runner",
                            "runtime_acceptance": ["Feature state changes authoritatively."],
                        },
                    ],
                    "completion_predicate": {
                        "operator": "all",
                        "model_self_report_is_authoritative": False,
                    },
                },
            },
        },
        "initial_exact_source_context": {
            "observations": [{"path": "Feature.java", "content": "class Feature {}"}]
        },
        "research_context": {"selected_facts": ["verified API fact"]},
        "host_grounding": {"schema_version": "mmm/host-owned-coder-grounding"},
        "rules": ["Use tools."],
    }
    return (
        {"role": "system", "content": "coder"},
        {"role": "user", "content": json.dumps(request)},
    )


def _request(batch):
    return json.loads(batch[-1]["content"])


def _distinct_messages(*, step_count: int = 3):
    messages = list(_messages(step_count=step_count))
    request = json.loads(messages[-1]["content"])
    steps = request["module"]["evidence_task"]["coder_execution_contract"]["implementation_steps"]
    for index, step in enumerate(steps, start=1):
        step["must_provide"] = [f"feature_ready_{index}"]
    messages[-1] = {"role": "user", "content": json.dumps(request)}
    return tuple(messages)


def test_production_whole_file_coder_does_not_recreate_retired_atomic_seams() -> None:
    from minecraft_mod_ai import custom_module_generator, model_router
    from minecraft_mod_ai.small_model_atomic_coder_execution import (
        assert_installed,
        install,
    )

    retired = (
        "_generate_coder_text",
        "_collect_initial_observations",
        "_materialize_owned_reuse_context",
    )
    assert all(not hasattr(custom_module_generator, name) for name in retired)

    install(
        custom_module_generator_module=custom_module_generator,
        model_router_module=model_router,
    )

    assert all(not hasattr(custom_module_generator, name) for name in retired)
    assert_installed(
        custom_module_generator_module=custom_module_generator,
        model_router_module=model_router,
    )


def test_distinct_state_transitions_become_one_model_batch_each() -> None:
    batches = atomicize_coder_messages(_distinct_messages(step_count=3))

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
        == "one_model_call_one_host_owned_state_transition"
        for batch in batches
    )


def test_coowned_constraint_steps_are_coalesced_into_one_state_transition() -> None:
    batches = atomicize_coder_messages(_messages(step_count=3))

    assert len(batches) == 1
    request = _request(batches[0])
    step = request["module"]["evidence_task"]["coder_execution_contract"]["step"]
    assert "Implement isolated behavior 1." in step["obligation"]
    assert "Implement isolated behavior 2." in step["obligation"]
    assert "Implement isolated behavior 3." in step["obligation"]
    assert step["must_provide"] == ["feature_ready"]
    assert request["atomic_execution"]["step_count"] == 1
    assert (
        request["atomic_execution"]["policy"]
        == "one_model_call_one_host_owned_state_transition"
    )


def test_atomic_batch_removes_siblings_but_preserves_required_canonical_context() -> None:
    batches = atomicize_coder_messages(_distinct_messages(step_count=3))

    first_text = batches[0][-1]["content"]
    assert "Implement isolated behavior 1." in first_text
    assert "Implement isolated behavior 2." not in first_text
    assert "Implement isolated behavior 3." not in first_text
    request = _request(batches[0])
    evidence = request["module"]["evidence_task"]
    contract = evidence["coder_execution_contract"]

    assert set(evidence) == {"task_id", "task_sha256", "coder_execution_contract"}
    assert "research_reuse_candidates" not in first_text
    assert "outer_copy" not in first_text
    assert contract["schema_version"] == "mmm/atomic-coder-step"
    assert contract["semantic_outcome"] == "Feature behaves exactly as approved."
    assert contract["engineering_worksheet"]["runtime_behavior"]["status"] == "applicable"
    assert contract["step"]["execution_checklist"] == [f"check-{index}" for index in range(8)]
    assert contract["step"]["target_refs"] == ["src/main/java/demo/Feature.java#Feature"]
    assert contract["step"]["consumes"] == ["core_ready"]
    assert contract["step"]["must_provide"] == ["feature_ready_1"]
    assert contract["targets"] == [
        {
            "path": "src/main/java/demo/Feature.java",
            "locator": "src/main/java/demo/Feature.java#Feature",
        }
    ]
    assert contract["protected_boundaries"]["writable_paths"] == [
        "src/main/java/demo/Feature.java",
        "src/test/java/demo/FeatureTest.java",
    ]
    assert [item["gate"] for item in contract["verification_plan"]] == [
        "target_compile",
        "observable_acceptance",
    ]
    assert contract["completion_predicate"]["model_self_report_is_authoritative"] is False
    assert contract["source_contract_sha256"] == "sha256:" + "b" * 64


def test_later_atomic_steps_do_not_receive_stale_pre_step_source_page() -> None:
    batches = atomicize_coder_messages(_distinct_messages(step_count=3))

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
    assert contract["schema_version"] == "mmm/atomic-coder-step"
    assert contract["step"] == {
        "index": 1,
        "count": 1,
        "sequence": 0,
        "obligation": "Implement isolated behavior 1.",
        "target_refs": ["src/main/java/demo/Feature.java#Feature"],
        "consumes": ["core_ready"],
        "must_provide": ["feature_ready"],
        "execution_checklist": [f"check-{index}" for index in range(8)],
        "done_when": "The isolated behavior is implemented and ready for host verification.",
    }
    assert "implementation_steps" not in contract


def test_implementation_request_without_contract_fails_closed() -> None:
    messages = list(_messages(step_count=1))
    request = json.loads(messages[-1]["content"])
    del request["module"]["evidence_task"]["coder_execution_contract"]
    messages[-1] = {"role": "user", "content": json.dumps(request)}

    with pytest.raises(AtomicCoderContractError, match="CODER_CONTRACT_LOWERING_FAILED"):
        atomicize_coder_messages(tuple(messages))


def test_versioned_or_stale_coder_schema_is_rejected() -> None:
    messages = list(_messages(step_count=1))
    request = json.loads(messages[-1]["content"])
    request["module"]["evidence_task"]["coder_execution_contract"]["schema_version"] = (
        "mmm/coder-execution-contract-v2"
    )
    messages[-1] = {"role": "user", "content": json.dumps(request)}

    with pytest.raises(AtomicCoderContractError, match="non-canonical coder execution schema"):
        atomicize_coder_messages(tuple(messages))


def test_malformed_sibling_step_cannot_be_silently_dropped() -> None:
    messages = list(_messages(step_count=2))
    request = json.loads(messages[-1]["content"])
    request["module"]["evidence_task"]["coder_execution_contract"]["implementation_steps"][1] = {
        "sequence": 1,
        "obligation": "",
    }
    messages[-1] = {"role": "user", "content": json.dumps(request)}

    with pytest.raises(AtomicCoderContractError, match="CODER_CONTRACT_LOWERING_FAILED"):
        atomicize_coder_messages(tuple(messages))


def test_unrelated_model_request_is_not_rewritten() -> None:
    messages = ({"role": "user", "content": "plain question"},)
    assert atomicize_coder_messages(messages) == (messages,)


def test_atomic_summary_aggregation_preserves_response_template():
    from types import SimpleNamespace

    from minecraft_mod_ai.model_response_templates import response_schema
    from minecraft_mod_ai.small_model_atomic_coder_execution import install

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert kwargs["response_schema"] == response_schema("coder_summary")
            return json.dumps({"summary": "Completed an isolated obligation."})

    custom = SimpleNamespace(
        _coder_project_context_budget=lambda *a, **k: 4096,
        _generate_coder_text=(
            lambda router, role, messages, *args, **kwargs:
            router.generate_text(role, messages, *args, **kwargs)
        ),
        _collect_initial_observations=lambda *a, **k: {},
        _materialize_owned_reuse_context=lambda *a, **k: {},
    )
    original_router_generate_text = Router.generate_text
    install(
        custom_module_generator_module=custom,
        model_router_module=SimpleNamespace(ModelRouter=Router),
    )
    assert Router.generate_text is original_router_generate_text
    result = custom._generate_coder_text(Router(),
        "coder",
        _distinct_messages(step_count=3),
        response_format="json",
        response_schema=response_schema("coder_summary"),
    )
    summary = json.loads(result)["summary"]
    assert summary.count("Completed an isolated obligation.") == 3
    assert "atomic step 3/3" in summary


def _two_target_capsule() -> TaskCapsule:
    anchors = (
        TaskAnchor(
            kind="symbol",
            path="src/main/java/demo/Feature.java",
            symbol="Feature",
            status="existing",
        ),
        TaskAnchor(
            kind="test",
            path="src/test/java/demo/FeatureTest.java",
            symbol="FeatureTest",
            status="existing",
        ),
    )
    return TaskCapsule(
        task_id="task_feature",
        module_kind="custom_java",
        primary_path=anchors[0].path,
        primary_symbol=anchors[0].symbol,
        anchors=anchors,
        reuse_action="fresh",
        required_gates=("target_compile",),
        task_sha256="sha256:" + "a" * 64,
        capsule_sha256="sha256:" + "b" * 64,
    )


def _atomic_scope_messages(target_ref: str, *, step_index: int = 2):
    return (
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "atomic_execution": {
                        "schema_version": "mmm/atomic-coder-step",
                        "step_index": step_index,
                        "step_count": 2,
                    },
                    "module": {
                        "module_id": "task_feature",
                        "kind": "custom_java",
                        "evidence_task": {
                            "task_id": "task_feature",
                            "coder_execution_contract": {
                                "schema_version": "mmm/atomic-coder-step",
                                "step": {
                                    "index": step_index,
                                    "count": 2,
                                    "target_refs": [target_ref],
                                },
                            },
                        },
                    },
                }
            ),
        },
    )


def test_atomic_step_capsule_exposes_only_step_owned_target() -> None:
    parent = _two_target_capsule()
    scoped, metadata = _atomic_request_scope(
        parent,
        _atomic_scope_messages(
            "src/test/java/demo/FeatureTest.java#FeatureTest"
        ),
    )

    assert metadata == {"step_index": 2, "step_count": 2}
    assert scoped.primary_path == "src/test/java/demo/FeatureTest.java"
    assert scoped.primary_symbol == "FeatureTest"
    assert scoped.writable_paths == ("src/test/java/demo/FeatureTest.java",)
    assert scoped.capsule_sha256 != parent.capsule_sha256

    with pytest.raises(TaskCapsuleContractError, match="MUTATION_TARGET_DRIFT"):
        bind_source_edit_arguments(
            {
                "operation": "replace_exact",
                "path": "src/main/java/demo/Feature.java",
                "old": "old",
                "new": "new",
            },
            scoped,
        )


def test_atomic_step_capsule_rejects_target_outside_parent_task() -> None:
    with pytest.raises(
        TaskCapsuleContractError,
        match="TASK_CAPSULE_ATOMIC_SCOPE_ESCAPE",
    ):
        _atomic_request_scope(
            _two_target_capsule(),
            _atomic_scope_messages(
                "src/main/java/demo/Foreign.java#Foreign",
            ),
        )



def test_direct_coder_no_longer_exports_legacy_observation_pager() -> None:
    from minecraft_mod_ai import custom_module_generator

    assert not hasattr(custom_module_generator, "_collect_initial_observations")
    assert not hasattr(custom_module_generator, "_observation_context_pages")

