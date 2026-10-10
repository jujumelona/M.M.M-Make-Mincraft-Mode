"""Regression: unresolved quality/runtime must not erase a successful build ZIP."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import work_graph


def _quality_plan(monkeypatch):
    # Isolate graph dependency semantics without invoking a model, Gradle or
    # the unrelated target resolver.
    monkeypatch.setattr(
        work_graph,
        "compile_production_routing",
        lambda proposal, *, modules: SimpleNamespace(
            route_by_module_id={},
            contract_sha256="sha256:" + "a" * 64,
            to_dict=lambda: {"routes": []},
        ),
    )
    proposal = SimpleNamespace(
        validate=lambda **kwargs: None,
        approval_hash="sha256:" + "1" * 64,
        calculate_hash=lambda: "sha256:" + "1" * 64,
        modules=(),
        assets=(),
        existing_input_sha256="",
        acceptance_tests=("Build the mod",),
        external_runtime_required=True,
        schema_version="mmm/complete-proposal-v2",
        game_design={
            "_production_contract": {
                "contract_sha256": "sha256:" + "2" * 64,
                "quality_dimension_catalog": [
                    {
                        "dimension_id": name,
                        "evidence_route_ref": f"evidence:{name}",
                    }
                    for name in ("correctness", "build", "research", "runtime")
                ],
            }
        },
    )
    return work_graph.build_production_work_plan(proposal)


def test_build_artifact_has_no_quality_or_runtime_success_dependency(monkeypatch):
    plan = _quality_plan(monkeypatch)
    by_id = {node.node_id: node for node in plan.nodes}
    assert by_id["package-build-artifact"].dependencies == ("validate-jar",)
    assert set(by_id["package-release"].dependencies) == {
        "package-build-artifact",
        "runtime-playtest",
        "validate-quality-correctness",
        "validate-quality-build",
        "validate-quality-research",
        "validate-quality-runtime",
    }


def test_missing_quality_and_runtime_still_package_build_but_block_release(
    monkeypatch, tmp_path,
):
    plan = _quality_plan(monkeypatch)
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "production.sqlite", proposal_hash=plan.proposal_hash
    )
    ledger.sync_plan(plan)
    for node_id in (
        "prepare-project",
        "validate-source",
        "build-project",
        "validate-source-final",
        "validate-jar",
    ):
        ledger.begin(node_id)
        ledger.succeed(node_id, {"status": "PASS"})
    ledger.fail("runtime-playtest", "Needs real runtime", input_required=True)
    for name in ("correctness", "build", "research", "runtime"):
        ledger.fail(
            "validate-quality-" + name,
            "Missing independent evidence",
            input_required=True,
        )

    ledger.begin("package-build-artifact")
    ledger.succeed(
        "package-build-artifact",
        {"status": "PASS", "release_ready": False},
    )
    assert ledger.task("package-build-artifact")["state"] == "succeeded"
    with pytest.raises(work_graph.WorkGraphError, match="incomplete dependencies"):
        ledger.begin("package-release")


def test_release_needs_runtime_and_all_quality_receipts(monkeypatch, tmp_path):
    plan = _quality_plan(monkeypatch)
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "production.sqlite", proposal_hash=plan.proposal_hash
    )
    ledger.sync_plan(plan)
    for node_id in (
        "prepare-project",
        "validate-source",
        "build-project",
        "validate-source-final",
        "validate-jar",
        "runtime-playtest",
        "validate-quality-correctness",
        "validate-quality-build",
        "validate-quality-research",
        "validate-quality-runtime",
        "package-build-artifact",
    ):
        ledger.begin(node_id)
        ledger.succeed(node_id, {"status": "PASS"})
    assert ledger.begin("package-release")["state"] == "running"
