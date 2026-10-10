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



def _evidenced_receipt(node_id: str) -> dict:
    """Use proof-shaped receipts so the dependency test never disables the verifier."""
    if node_id in {"validate-source", "validate-source-final"}:
        return {
            "status": "PASS",
            "checks_run": 2,
            "project_manifest": "sha256:" + "a" * 64,
        }
    if node_id == "build-project":
        return {
            "status": "PASS",
            "build": {
                "status": "PASS",
                "commands": [
                    {"name": "build", "exit_code": 0, "timed_out": False}
                ],
                "artifact_receipt": {"sha256": "sha256:" + "b" * 64},
            },
            "final_build_receipt": {
                "status": "PASS",
                "production_jar": "PASS",
                "artifact_sha256": "sha256:" + "b" * 64,
                "toolchain_attested": True,
            },
        }
    if node_id == "validate-jar":
        return {
            "status": "PASS",
            "checks_run": 2,
            "jar_sha256": "sha256:" + "b" * 64,
        }
    if node_id == "runtime-playtest":
        return {
            "status": "PASS",
            "runtime": {"server": {"server_running": True}},
            "playtest": {
                "status": "PASS",
                "interaction_count": 1,
                "assertion_count": 1,
            },
            "visual": {"status": "PASS"},
        }
    if node_id == "package-build-artifact":
        return {
            "status": "PASS",
            "build_bundle_zip": "/tmp/fixture-build-artifact.zip",
            "release_ready": False,
            "unresolved_gates": ["quality:runtime"],
        }
    if node_id.startswith("validate-quality-"):
        dimension = node_id.removeprefix("validate-quality-")
        return {
            "status": "PASS",
            "dimension_id": dimension,
            "receipt_id": "test-" + dimension,
            "receipt_sha256": "sha256:" + "c" * 64,
        }
    return {"status": "PASS"}


def _succeed_with_evidence(ledger, node_id: str) -> None:
    ledger.begin(node_id)
    ledger.succeed(node_id, _evidenced_receipt(node_id))


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
        _succeed_with_evidence(ledger, node_id)
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
        _evidenced_receipt("package-build-artifact"),
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
        _succeed_with_evidence(ledger, node_id)
    assert ledger.begin("package-release")["state"] == "running"


def test_verified_nodes_reject_fake_pass_receipts(monkeypatch, tmp_path):
    plan = _quality_plan(monkeypatch)
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "fake-pass.sqlite", proposal_hash=plan.proposal_hash
    )
    ledger.sync_plan(plan)
    _succeed_with_evidence(ledger, "prepare-project")
    ledger.begin("validate-source")
    from minecraft_mod_ai.verifier_receipt_truth_contract import VerifierReceiptTruthError
    with pytest.raises(VerifierReceiptTruthError, match="VERIFIER_RECEIPT_MISSING"):
        ledger.succeed("validate-source", {"status": "PASS"})
