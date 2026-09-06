from __future__ import annotations

from minecraft_mod_ai import evidence_execution_contract as execution
from minecraft_mod_ai import research_derived_requirements as derivation
from minecraft_mod_ai.plan_collect_all_linker import collect_plan_link_issues


def _task(
    task_id: str,
    *,
    provides: list[str],
    anchors: list[dict[str, object]],
    depends_on: list[str] | None = None,
    semantic_outcome: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "semantic_outcome": semantic_outcome or task_id,
        "gap_refs": ["gap_demo"],
        "requirement_refs": ["req_demo"],
        "target_cell": {},
        "owned_anchors": anchors,
        "reuse_refs": [],
        "consumes": [],
        "provides": provides,
        "depends_on": depends_on or [],
        "conditional_predicates": [],
        "required_gates": ["source_static_validation", "target_compile"],
        "acceptance": ["observable outcome passes"],
        "done_predicate": {"operator": "all", "checks": ["required_gates_passed"]},
        "impact_probes": [],
        "state": "pending",
        "task_sha256": "sha256:original",
    }


def test_execution_lowering_binds_runtime_and_keeps_non_source_steps_typed(monkeypatch):
    monkeypatch.setattr(execution, "validate_evidence_first_plan", lambda _plan: None)
    registry = _task(
        "task_registry_identity",
        provides=["registry_id:demo"],
        anchors=[
            {
                "kind": "registry_id",
                "locator": "registry:demo:block/registry_identity",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "main",
            }
        ],
    )
    runtime = _task(
        "task_runtime_scenario",
        provides=["capability:space_travel"],
        anchors=[
            {
                "kind": "test",
                "locator": "src/test/java/example/SpaceTravelTest.java#SpaceTravelTest",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "test",
            }
        ],
    )
    resource = _task(
        "task_resource_binding",
        provides=["resource:space_travel"],
        anchors=[
            {
                "kind": "resource",
                "locator": "resource:demo:space_travel/resource_binding",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": ":",
                "source_set": "resources",
            }
        ],
    )
    plan = {
        "plan_sha256": "sha256:semantic",
        "ownership_context": {
            "source_root": "src/main/java",
            "namespace": "example.mod",
            "extension": "java",
            "module_id": ":",
            "source_set": "main",
        },
        "reuse_decisions": [
            {"requirement_ref": "req_demo", "action": "fresh"},
        ],
        "tasks": [registry, resource, runtime],
    }
    canonical_handoff = {
        "handoff_sha256": "canonical-handoff",
        "production_modules": [],
        "asset_requests": [
            {
                "asset_request_id": "asset-resource",
                "task_ref": "task_resource_binding",
                "reuse_action": "fresh",
            }
        ],
    }

    lowered = execution.execution_plan(plan)
    handoff = execution.execution_handoff(plan, canonical_handoff, lowered)
    by_id = {item["task_id"]: item for item in lowered["tasks"]}

    runtime_lowered = by_id["task_runtime_scenario"]
    assert runtime_lowered["execution_role"] == "production_with_verification"
    assert {anchor["kind"] for anchor in runtime_lowered["owned_anchors"]} == {
        "symbol",
        "test",
    }
    assert any(
        binding["task_ref"] == "task_runtime_scenario"
        for binding in handoff["production_modules"]
    )

    registry_lowered = by_id["task_registry_identity"]
    assert registry_lowered["execution_role"] == "production"
    assert {anchor["kind"] for anchor in registry_lowered["owned_anchors"]} == {
        "registry_id"
    }
    assert "source_static_validation" not in registry_lowered["required_gates"]
    assert "target_compile" not in registry_lowered["required_gates"]
    assert any(
        binding["task_ref"] == "task_registry_identity"
        for binding in handoff["production_modules"]
    )

    resource_lowered = by_id["task_resource_binding"]
    assert resource_lowered["execution_role"] == "resource"
    assert {anchor["kind"] for anchor in resource_lowered["owned_anchors"]} == {"resource"}
    assert "source_static_validation" not in resource_lowered["required_gates"]
    assert "target_compile" not in resource_lowered["required_gates"]

    issues = collect_plan_link_issues(lowered, handoff)
    assert [issue.to_dict() for issue in issues] == []


class _FacetRouter:
    def __init__(self) -> None:
        self.calls = 0

    def generate_text(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("deterministic host facet closure must not call the model")


class _NoCallRouter:
    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("generic unrelated evidence must not trigger a model turn")


def _derivation_plan() -> dict[str, object]:
    return {
        "tasks": [
            {
                "task_id": "task_demo",
                "requirement_refs": ["req_demo"],
                "owned_anchors": [{"kind": "symbol"}, {"kind": "test"}],
                "semantic_outcome": "Persist and reload travel state",
                "acceptance": ["Travel is observable"],
            }
        ],
        "request_catalog": {
            "prompt_sha256": "sha256:prompt",
            "requirements": [
                {
                    "requirement_id": "req_demo",
                    "statement": "travel to another world",
                    "capability": "capability:space_travel",
                    "implementation_capabilities": ["network.action_sync"],
                }
            ],
        },
    }


def _derive(router, synthetic_platform_lock, *, research_brief):
    return derivation.derive_research_requirements(
        router,
        prompt="travel to another world",
        evidence_plan=_derivation_plan(),
        research_brief=research_brief,
        technical_evidence={},
        game_design={
            "_platform_selection": {
                "source": "platform_resolver",
                "target": synthetic_platform_lock.to_dict(),
            }
        },
    )


def test_research_derivation_closes_facets_without_model_planning(
    monkeypatch, synthetic_platform_lock
):
    monkeypatch.setattr(
        derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None
    )
    router = _FacetRouter()
    ledger = _derive(
        router,
        synthetic_platform_lock,
        research_brief={
            "source_id": "research:runtime",
            "url": "https://example.org/test-fixture/runtime",
            "requirement_ref": "req_demo",
            "claim": "verification evidence: runtime transition is externally observable",
        },
    )
    decisions = ledger["facet_decisions"]
    assert router.calls == 0
    assert ledger["model_call_policy"]["actual_calls_including_retries"] == 0
    assert ledger["model_call_policy"]["unit"] == "none"
    assert ledger["host_template"]["model_generated_planning_json"] is False
    assert len(decisions) == len(derivation.FACETS)
    assert all(item["disposition"] != "unresolved" for item in decisions)
    derived = [item for item in decisions if item["disposition"] == "derived"]
    assert [item["facet"] for item in derived] == ["server_network_authority"]
    assert derived[0]["parent_requirement_ref"] == "req_demo"
    assert derived[0]["provenance_role"] == "logically_derived"
    assert derived[0]["owner_task_ref"] == "task_demo"
    assert derived[0]["acceptance"]
    assert derived[0]["implementation_obligations"]


def test_generic_unbound_evidence_does_not_manufacture_unresolved_facets(
    monkeypatch, synthetic_platform_lock
):
    monkeypatch.setattr(
        derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None
    )
    ledger = _derive(
        _NoCallRouter(),
        synthetic_platform_lock,
        research_brief={
            "source_id": "research:generic",
            "claim": "general platform metadata is available",
        },
    )
    assert ledger["model_call_policy"]["actual_calls_including_retries"] == 0
    assert all(
        item["disposition"] != "unresolved" for item in ledger["facet_decisions"]
    )


def test_research_absence_is_closed_by_host_template_not_model_failure(
    monkeypatch, synthetic_platform_lock
):
    monkeypatch.setattr(
        derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None
    )
    router = _FacetRouter()
    ledger = _derive(
        router,
        synthetic_platform_lock,
        research_brief={
            "source_id": "research:persistence",
            "url": "https://example.org/test-fixture/persistence",
            "requirement_ref": "req_demo",
            "claim": "persistence reload evidence for travel state is incomplete",
        },
    )
    assert router.calls == 0
    assert ledger["model_call_policy"]["actual_calls_including_retries"] == 0
    assert ledger["model_call_policy"]["research_absence_behavior"] == (
        "continue_with_host_template"
    )
    assert all(
        item["disposition"] != "unresolved" for item in ledger["facet_decisions"]
    )
    assert ledger["host_closure"]["required_facets_closed"] == 1
