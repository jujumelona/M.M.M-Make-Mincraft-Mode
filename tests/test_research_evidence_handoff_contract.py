from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import custom_module_generator, research_ledger
from minecraft_mod_ai import research_evidence_handoff_contract as contract


class _Quality:
    def to_dict(self) -> dict[str, float]:
        return {"correctness": 0.91, "maintainability": 0.87}


def test_runtime_installs_reference_policy_on_live_coder_selector() -> None:
    selector = research_ledger.select_module_research_context

    assert getattr(selector, contract._MARKER, False) is True
    assert custom_module_generator.select_module_research_context is selector


def test_reference_context_never_grants_source_reuse_authority() -> None:
    base = {
        "schema_version": "mmm/module-research-context-v1",
        "ledger_fact_count": 2,
        "selected_record_count": 1,
        "selected_fact_count": 2,
        "selected_facts_sha256": "sha256:original",
        "records": [
            {
                "record_id": "record:1",
                "source_id": "candidate:example",
                "source_type": "ecosystem_candidate",
                "fact_ids": ["fact:1", "fact:2"],
                "fields": {
                    "/revision_sha": "deadbeef",
                    "/license/id": "MIT",
                    "/reuse_status": "candidate",
                },
            }
        ],
        "policy": {"facts_are_data_not_instructions": True},
    }

    result = contract._reference_only_context(base, byte_budget=4096)

    assert result["records"] == base["records"]
    assert result["policy"]["reference_only"] is True
    assert result["policy"]["source_reuse_authority"] == "approved_reuse_context_only"
    assert result["policy"]["pinned_license_provenance_required_for_source_reuse"] is True


def test_reusable_evidence_preserves_quality_plan_and_ranking_metadata() -> None:
    evidence = SimpleNamespace(
        evidence_id="repo:example",
        source_type="repository_symbol",
        path="src/main/java/example/Example.java",
        sha256="sha256:abc",
        start_line=10,
        end_line=30,
        symbols=("Example", "register"),
        plan_steps={"step-2", "step-1"},
        metrics={"semantic": 0.8, "quality": 0.9},
        quality=_Quality(),
        bestfit_score=0.93,
        graph_hop=1,
        algorithmic_plan="resolve API -> bind registry -> verify behavior",
        text="public final class Example { static void register() {} }",
    )
    context = SimpleNamespace(evidence={"repo:example": evidence})

    result = contract._full_reusable_evidence(context)

    assert result[0]["evidence_id"] == "repo:example"
    assert result[0]["plan_steps"] == ["step-1", "step-2"]
    assert result[0]["bestfit_score"] == 0.93
    assert result[0]["quality"]["correctness"] == 0.91
    assert result[0]["metrics"]["semantic"] == 0.8
    assert "bind registry" in result[0]["algorithmic_plan"]


def test_diagnostic_queries_are_narrowed_to_actual_failure_symbols_and_paths() -> None:
    diagnostic = {
        "files": ["src/main/java/example/GravityHandler.java"],
        "symbols": ["ServerPlayerEntity", "setVelocity"],
        "exceptions": ["NoSuchMethodError"],
        "tasks": [":compileJava"],
        "messages": [
            "GravityHandler.java:42 cannot find symbol method setVelocity(double,double,double)"
        ],
    }

    queries = contract._diagnostic_queries(diagnostic)

    assert len(queries) == 2
    assert any("ServerPlayerEntity" in query for query in queries)
    assert any("setVelocity" in query for query in queries)
    assert any("GravityHandler.java" in query for query in queries)
    assert any(":compileJava" in query for query in queries)
