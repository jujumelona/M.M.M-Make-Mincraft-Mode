from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import custom_module_generator, research_ledger
from minecraft_mod_ai import repair_approved_reuse_context as repair_reuse
from minecraft_mod_ai import research_coder_repair_reuse as reuse_hardener
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


def test_reference_candidate_cannot_authorize_repair_source_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        repair_reuse,
        "_load_approved_reuse_plan",
        lambda _root: {
            "capabilities": [
                {
                    "mode": "reference",
                    "capability": "gravity",
                    "donor": {"repository": "example/reference-only"},
                }
            ]
        },
    )

    def forbidden_materialization(*_args, **_kwargs):
        raise AssertionError("reference-only evidence must never materialize as donor source")

    monkeypatch.setattr(
        repair_reuse,
        "materialize_source_slices",
        forbidden_materialization,
    )

    result = repair_reuse.build_approved_repair_reuse_context(
        tmp_path,
        {"symbols": ["GravityHandler"]},
    )

    assert result is None


def test_verified_repair_donor_reaches_bounded_coder_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = {
        "capabilities": [
            {
                "mode": "source_transplant",
                "capability": "gravity",
                "donor": {"repository": "example/verified"},
            }
        ]
    }
    monkeypatch.setattr(repair_reuse, "_load_approved_reuse_plan", lambda _root: plan)
    monkeypatch.setattr(
        repair_reuse,
        "materialize_source_slices",
        lambda _root, _plan: {
            "schema_version": "mmm/reuse-materialization-v1",
            "count": 1,
            "donors": [
                {
                    "repository": "example/verified",
                    "commit_sha": "a" * 40,
                    "license_id": "MIT",
                    "capability": "gravity",
                    "files": [
                        {
                            "path": "/approved/GravityHandler.java",
                            "source_path": "src/main/java/example/GravityHandler.java",
                            "sha256": "sha256:" + "b" * 64,
                            "symbols": ["GravityHandler", "applyGravity"],
                        }
                    ],
                }
            ],
        },
    )

    class _Service:
        def __init__(self, *, workspace_root: Path) -> None:
            assert workspace_root == tmp_path.resolve()

        def read_reuse_source(
            self,
            project_root: str,
            path: str,
            *,
            limit_bytes: int,
        ) -> dict[str, object]:
            assert project_root == "."
            assert path == "/approved/GravityHandler.java"
            assert 1 <= limit_bytes <= 6 * 1024
            return {
                "repository": "example/verified",
                "commit_sha": "a" * 40,
                "license_id": "MIT",
                "capability": "gravity",
                "path": path,
                "sha256": "sha256:" + "b" * 64,
                "content": "public final class GravityHandler { void applyGravity() {} }",
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(repair_reuse, "ProductionToolService", _Service)

    result = repair_reuse.build_approved_repair_reuse_context(
        tmp_path,
        {
            "files": ["src/main/java/generated/GravityHandler.java"],
            "symbols": ["GravityHandler", "applyGravity"],
        },
        byte_budget=4096,
    )

    assert result is not None
    assert result["status"] == "APPROVED"
    assert result["policy"]["source_reuse_authority"] == "verified_reuse_plan_only"
    assert result["snippets"][0]["repository"] == "example/verified"
    assert "applyGravity" in result["snippets"][0]["content"]
    assert result["bytes_used"] <= result["byte_budget"]


def test_unverified_repair_donor_is_rejected_before_source_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        repair_reuse,
        "_load_approved_reuse_plan",
        lambda _root: {
            "capabilities": [
                {
                    "mode": "adapt",
                    "capability": "gravity",
                    "donor": {"repository": "example/unverified"},
                }
            ]
        },
    )

    def reject(*_args, **_kwargs):
        raise repair_reuse.SourceTransplantError("proof level is not verified")

    monkeypatch.setattr(repair_reuse, "materialize_source_slices", reject)

    with pytest.raises(repair_reuse.RepairReuseContextError, match="verification failed"):
        repair_reuse.build_approved_repair_reuse_context(
            tmp_path,
            {"symbols": ["GravityHandler"]},
        )


def test_repair_context_wrapper_hands_verified_donor_to_repair_coder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _RepairEngine:
        def _context(self, root: Path, evidence: dict[str, object]) -> dict[str, object]:
            return {"diagnostics_files": (), "rag": {"hits": []}}

    monkeypatch.setattr(
        reuse_hardener,
        "_diagnostic_signature_payload",
        lambda _evidence: {
            "files": ["src/main/java/generated/GravityHandler.java"],
            "symbols": ["GravityHandler"],
            "exceptions": [],
            "tasks": [":compileJava"],
            "messages": ["cannot find symbol GravityHandler"],
        },
    )
    approved = {
        "schema_version": "mmm/approved-repair-reuse-context-v1",
        "status": "APPROVED",
        "snippets": [{"content": "class GravityHandler {}"}],
    }
    monkeypatch.setattr(
        repair_reuse,
        "build_approved_repair_reuse_context",
        lambda _root, _diagnostic: approved,
    )

    repair_reuse.install(SimpleNamespace(RepairEngine=_RepairEngine))
    result = _RepairEngine()._context(tmp_path, {"build": {}, "diagnostics": {}})

    assert result["approved_reuse_context"] == approved
    assert result["retrieval_policy"]["verified_donor_required_for_source_reuse"] is True
    assert result["retrieval_policy"]["reference_evidence_cannot_authorize_source_reuse"] is True
    assert result["retrieval_policy"]["approved_repair_donor_available"] is True
