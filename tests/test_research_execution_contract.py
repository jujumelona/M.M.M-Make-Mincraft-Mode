from __future__ import annotations

import json

import pytest

from minecraft_mod_ai import evidence_execution_contract as execution
from minecraft_mod_ai import research_derived_requirements as derivation
from minecraft_mod_ai.plan_collect_all_linker import collect_plan_link_issues


def _task(
    task_id: str,
    *,
    provides: list[str],
    anchors: list[dict[str, object]],
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "semantic_outcome": task_id,
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
    assert {anchor["kind"] for anchor in runtime_lowered["owned_anchors"]} == {"symbol", "test"}
    assert any(
        binding["task_ref"] == "task_runtime_scenario"
        for binding in handoff["production_modules"]
    )

    registry_lowered = by_id["task_registry_identity"]
    assert registry_lowered["execution_role"] == "production"
    assert {anchor["kind"] for anchor in registry_lowered["owned_anchors"]} == {"registry_id"}
    assert "source_static_validation" not in registry_lowered["required_gates"]
    assert "target_compile" not in registry_lowered["required_gates"]
    assert any(
        binding["task_ref"] == "task_registry_identity"
        for binding in handoff["production_modules"]
    )

    resource_lowered = by_id["task_resource_binding"]
    assert resource_lowered["execution_role"] == "resource"
    assert "source_static_validation" not in resource_lowered["required_gates"]
    assert "target_compile" not in resource_lowered["required_gates"]

    issues = collect_plan_link_issues(lowered, handoff)
    assert [issue.to_dict() for issue in issues] == []


class _FacetRouter:
    def __init__(self, unresolved: bool = False) -> None:
        self.unresolved = unresolved
        self.calls = 0

    def generate_text(self, _role, messages, **_kwargs):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        evidence_ref = payload["evidence_catalog"][0]["evidence_ref"]
        facets = []
        for facet in derivation.FACETS:
            disposition = (
                "unresolved"
                if self.unresolved and facet == "persistence_reload"
                else "not_applicable"
            )
            entry = {
                "facet": facet,
                "disposition": disposition,
                "statement": "",
                "rationale": "evidence does not require this facet",
                "evidence_refs": [],
                "acceptance": [],
                "implementation_obligations": [],
            }
            if facet == "verification_testing" and not self.unresolved:
                entry.update(
                    {
                        "disposition": "derived",
                        "statement": "verify the requested runtime transition externally",
                        "rationale": "the runtime evidence exposes an observable transition",
                        "evidence_refs": [evidence_ref],
                        "acceptance": ["the transition passes an external runtime check"],
                        "implementation_obligations": ["add a GameTest covering the transition"],
                    }
                )
            facets.append(entry)
        return json.dumps({"facets": facets})


class _NoCallRouter:
    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("generic unrelated evidence must not trigger a model turn")


def _derivation_plan() -> dict[str, object]:
    return {
        "request_catalog": {
            "prompt_sha256": "sha256:prompt",
            "requirements": [
                {
                    "requirement_id": "req_demo",
                    "statement": "travel to another world",
                    "capability": "capability:space_travel",
                }
            ],
        }
    }


def test_research_derivation_requires_traceable_evidence_and_one_requirement_turn(monkeypatch):
    monkeypatch.setattr(derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None)
    router = _FacetRouter()
    ledger = derivation.derive_research_requirements(
        router,
        prompt="travel to another world",
        evidence_plan=_derivation_plan(),
        research_brief={
            "source_id": "research:runtime",
            "requirement_ref": "req_demo",
            "claim": "verification evidence: runtime transition is externally observable",
        },
        technical_evidence={},
        game_design={},
    )
    decisions = ledger["facet_decisions"]
    assert router.calls == 1
    assert ledger["model_call_policy"]["actual_calls"] == 1
    assert len(decisions) == len(derivation.FACETS)
    derived = [item for item in decisions if item["disposition"] == "derived"]
    assert len(derived) == 1
    assert derived[0]["parent_requirement_ref"] == "req_demo"
    assert derived[0]["provenance_role"] == "logically_derived"
    assert derived[0]["evidence_refs"]
    assert derived[0]["acceptance"]
    assert derived[0]["implementation_obligations"]


def test_generic_unbound_evidence_does_not_manufacture_unresolved_facets(monkeypatch):
    monkeypatch.setattr(derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None)
    ledger = derivation.derive_research_requirements(
        _NoCallRouter(),
        prompt="travel to another world",
        evidence_plan=_derivation_plan(),
        research_brief={
            "source_id": "research:generic",
            "claim": "general platform metadata is available",
        },
        technical_evidence={},
        game_design={},
    )
    assert ledger["model_call_policy"]["actual_calls"] == 0
    assert all(
        item["disposition"] != "unresolved"
        for item in ledger["facet_decisions"]
    )


def test_research_derivation_fails_closed_on_relevant_unresolved_facet(monkeypatch):
    monkeypatch.setattr(derivation, "validate_evidence_first_plan", lambda _plan, prompt=None: None)
    with pytest.raises(derivation.ResearchRequirementError, match="could not close"):
        derivation.derive_research_requirements(
            _FacetRouter(unresolved=True),
            prompt="travel to another world",
            evidence_plan=_derivation_plan(),
            research_brief={
                "source_id": "research:persistence",
                "requirement_ref": "req_demo",
                "claim": "persistence reload evidence for travel state is incomplete",
            },
            technical_evidence={},
            game_design={},
        )
