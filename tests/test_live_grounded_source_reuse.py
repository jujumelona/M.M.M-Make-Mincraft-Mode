from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import source_transplant
from minecraft_mod_ai.evidence_first_planning import _validated_external_reuse
from minecraft_mod_ai.grounded_source_reuse import build_repository_reuse_plan
from minecraft_mod_ai.reuse_proof_executor import ReuseProofReceipt
from minecraft_mod_ai.source_transplant import (
    DonorFile,
    DonorSlice,
    donor_closure_sha256,
)

_PATH = "src/main/java/donor/TradeEngine.java"


def _design() -> dict:
    return {
        "_pre_retrieval_plan": {
            "plan_sha256": "sha256:" + "1" * 64,
            "capability_graph": {
                "nodes": ["trade.transaction"],
                "edges": [],
                "sources": [
                    {
                        "capability": "trade.transaction",
                        "source": "request_catalog",
                    }
                ],
                "search_terms": [
                    {
                        "capability": "trade.transaction",
                        "terms": ["trade transaction engine"],
                    }
                ],
            },
        },
        "_platform_selection": {
            "target": {"minecraft_version": "1.21.1", "loader": "fabric"}
        },
        "_pre_design_research": {
            "domain_notes": [
                {
                    "grounded_evidence_cards": [
                        {
                            "source_id": "github:owner/trade-mod",
                            "source_url": "https://github.com/owner/trade-mod",
                            "source_title": "Trade transaction engine",
                            "exact_excerpt": "Server-owned atomic trade transaction code",
                            "page_ref": "page:trade",
                        }
                    ]
                }
            ]
        },
    }


def _donor() -> DonorSlice:
    source = b"package donor; public final class TradeEngine {}\n"
    return DonorSlice(
        capability="trade.transaction",
        repository="owner/trade-mod",
        commit_sha="a" * 40,
        license_id="MIT",
        source_url="https://github.com/owner/trade-mod",
        target_compatibility="exact",
        files=(
            DonorFile(
                path=_PATH,
                blob_sha="b" * 40,
                sha256="sha256:" + hashlib.sha256(source).hexdigest(),
                size_bytes=len(source),
                symbols=("TradeEngine",),
            ),
        ),
        seed_files=(_PATH,),
        source_symbols=("TradeEngine",),
        required_dependencies=(),
        donor_tests=(),
        confidence=0.95,
        closure_complete=True,
    )


def _proof(*, verified: bool) -> ReuseProofReceipt:
    return ReuseProofReceipt(
        candidate_id="owner/trade-mod@" + "a" * 40,
        capability="trade.transaction",
        commit_sha="a" * 40,
        closure_hash=donor_closure_sha256(_donor()),
        proof_level="COMPILE_VERIFIED" if verified else "MATERIALIZED",
        compile_passed=verified,
        tests_passed=False,
        unresolved_symbols=(),
        missing_resources=(),
        adaptations_applied=(),
        verified_capabilities=("trade.transaction",) if verified else (),
        residual_capabilities=() if verified else ("trade.transaction",),
        authoritative_compile=verified,
        verified_artifacts=(_PATH,) if verified else (),
    )


def test_live_bridge_emits_only_authoritatively_compiled_code_donor(tmp_path) -> None:
    seen: dict[str, object] = {}

    def inspect(**kwargs):
        seen["repository"] = kwargs["repository"]
        seen["capability"] = kwargs["capability"]
        return _donor()

    def prove(donor, **kwargs):
        seen["proof_donor"] = donor.repository
        seen["run_tests"] = kwargs["run_tests"]
        return _proof(verified=True)

    plan = build_repository_reuse_plan(
        _design(),
        discovery_client=SimpleNamespace(),
        donor_inspector=inspect,
        proof_executor=prove,
        target_workspace=tmp_path,
    )

    decision = plan["capabilities"][0]
    assert seen == {
        "repository": "owner/trade-mod",
        "capability": "trade.transaction",
        "proof_donor": "owner/trade-mod",
        "run_tests": True,
    }
    assert decision["mode"] == "source_transplant"
    assert decision["donor"]["files"][0]["path"] == _PATH
    assert decision["proof_receipt"]["verified_artifacts"] == [_PATH]
    assert plan["proof_receipts"][0]["status"] == "verified_code_reuse"


def test_live_bridge_falls_back_to_fresh_when_code_proof_is_not_authoritative(tmp_path) -> None:
    plan = build_repository_reuse_plan(
        _design(),
        discovery_client=SimpleNamespace(),
        donor_inspector=lambda **_kwargs: _donor(),
        proof_executor=lambda _donor, **_kwargs: _proof(verified=False),
        target_workspace=tmp_path,
    )

    assert plan["capabilities"][0]["mode"] == "fresh"
    assert "donor" not in plan["capabilities"][0]
    assert plan["proof_receipts"][0]["status"] == "rejected"


def test_planir_rejects_pinned_donor_without_executable_proof() -> None:
    raw = {
        "capability": "trade.transaction",
        "mode": "source_transplant",
        "source_id": "host-donor:owner/trade-mod@" + "a" * 40,
        "donor": _donor().to_dict(),
    }
    target = {
        "coordinates": {"minecraft_version": "1.21.1", "loader": "fabric"}
    }

    assert not _validated_external_reuse(
        raw,
        capability="trade.transaction",
        target=target,
    )
    raw["proof_receipt"] = _proof(verified=True).to_dict()
    assert _validated_external_reuse(
        raw,
        capability="trade.transaction",
        target=target,
    )


def test_selected_plan_refetches_exact_donor_bytes_and_preserves_symbols(
    monkeypatch,
    tmp_path,
) -> None:
    donor = _donor()
    expected = b"package donor; public final class TradeEngine {}\n"

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(source_transplant, "_github_client", lambda _token: Client())
    monkeypatch.setattr(
        source_transplant,
        "_fetch_blob_bytes",
        lambda _client, repository, blob_sha: (
            expected
            if repository == donor.repository and blob_sha == donor.files[0].blob_sha
            else b""
        ),
    )
    plan = {
        "capabilities": [
            {
                "capability": donor.capability,
                "mode": "source_transplant",
                "donor": donor.to_dict(),
                "proof_receipt": _proof(verified=True).to_dict(),
            }
        ]
    }

    receipt = source_transplant.materialize_source_slices(tmp_path, plan)

    file_receipt = receipt["donors"][0]["files"][0]
    assert receipt["count"] == 1
    assert file_receipt["symbols"] == ["TradeEngine"]
    assert file_receipt["source_path"] == _PATH
    assert file_receipt["blob_sha"] == donor.files[0].blob_sha
    assert file_receipt["sha256"] == donor.files[0].sha256
    assert Path(file_receipt["path"]).read_bytes() == expected


def test_reuse_proof_cannot_be_replayed_after_blob_manifest_changes() -> None:
    donor = _donor()
    decision = {
        "capability": donor.capability,
        "mode": "source_transplant",
        "source_id": "host-donor:owner/trade-mod@" + "a" * 40,
        "donor": donor.to_dict(),
        "proof_receipt": _proof(verified=True).to_dict(),
    }
    decision["donor"]["files"][0]["blob_sha"] = "d" * 40

    assert not _validated_external_reuse(
        decision,
        capability="trade.transaction",
        target={"coordinates": {"minecraft_version": "1.21.1", "loader": "fabric"}},
    )


def test_invalid_reuse_proof_creates_no_local_donor_paths(tmp_path) -> None:
    donor = _donor()
    plan = {
        "capabilities": [
            {
                "capability": donor.capability,
                "mode": "source_transplant",
                "donor": donor.to_dict(),
                "proof_receipt": {
                    **_proof(verified=True).to_dict(),
                    "closure_hash": "sha256:" + "0" * 64,
                },
            }
        ]
    }

    with pytest.raises(
        source_transplant.SourceTransplantError,
        match="closure hash",
    ):
        source_transplant.materialize_source_slices(tmp_path, plan)

    assert not (tmp_path / ".minecraft_ai" / "reuse" / "donors").exists()
