from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from minecraft_mod_ai.spec import SpecValidationError, canonical_json
from minecraft_mod_ai.technology_radar import (
    _assess_technology_candidate_with_receipt_key,
    _seal_technology_receipt,
    _technology_candidate_snapshot_sha256,
    assess_technology_candidate,
    assess_technology_compatibility,
    build_technology_radar,
)

_RECEIPT_KEY = b"mmm-test-receipt-key-32-bytes-minimum"


def _requirement(radar: dict[str, object], kind: str) -> dict[str, object]:
    requirements = radar["requirements"]
    assert isinstance(requirements, list)
    return next(item for item in requirements if item["capability_kind"] == kind)


def _complete_candidate(
    requirement: dict[str, object],
    *,
    candidate_id: str = "reviewed-candidate",
) -> dict[str, object]:
    target = dict(requirement["target"])
    target_evidence = _official_target_evidence(dict(target))
    target["bridge_verified"] = True
    required_tests = requirement["required_tests"]
    assert isinstance(required_tests, list)
    requirement_id = str(requirement["requirement_id"])
    revision = "a" * 40
    artifact_sha256 = "sha256:" + "b" * 64
    evidence_sha256 = "sha256:" + "c" * 64
    candidate: dict[str, object] = {
        "candidate_id": candidate_id,
        "capability_kind": requirement["capability_kind"],
        "topology": "local_sidecar",
        "revision": revision,
        "artifact_sha256": artifact_sha256,
        "evidence_sha256": evidence_sha256,
        "formats": ["safetensors"],
        "licenses": {
            "code": {"id": "MIT", "reviewed": True, "use_allowed": True},
            "model": {
                "id": "Apache-2.0",
                "reviewed": True,
                "use_allowed": True,
            },
            "data": {
                "id": "CC-BY-4.0",
                "reviewed": True,
                "use_allowed": True,
            },
        },
        "dataset_provenance": {
            "sources": ["immutable-dataset-receipt"],
            "verified": True,
        },
        "official_target_evidence": target_evidence,
        "compatibility": target,
        "authority": {
            "game_state_mutation": "server_only",
            "client_messages_validated": True,
        },
        "runtime": {"device": "gpu", "supports_offline": True},
        "benchmarks": {
            "measured_on": "declared-test-gpu",
            "p50_latency_ms": 50,
            "p95_latency_ms": 80,
            "peak_memory_mb": 2048,
            "startup_ms": 900,
            "concurrency": 2,
        },
        "privacy": {
            "raw_input_leaves_device": False,
            "retention_policy": "none",
            "deletion_supported": True,
        },
        "fallback": {
            "description": requirement["deterministic_fallback"],
            "deterministic": True,
        },
        "maintenance": {"last_modified": "2024-01-01T00:00:00Z"},
    }
    candidate_snapshot_sha256 = _technology_candidate_snapshot_sha256(candidate)
    candidate["tests"] = {
        name: _technology_test_receipt(
            name,
            requirement_id,
            candidate_id=candidate_id,
            revision=revision,
            artifact_sha256=artifact_sha256,
            evidence_sha256=evidence_sha256,
            candidate_snapshot_sha256=candidate_snapshot_sha256,
        )
        for name in required_tests
    }
    candidate["fallback"]["test_receipt"] = _technology_test_receipt(
        "deterministic_fallback",
        requirement_id,
        candidate_id=candidate_id,
        revision=revision,
        artifact_sha256=artifact_sha256,
        evidence_sha256=evidence_sha256,
        candidate_snapshot_sha256=candidate_snapshot_sha256,
    )
    return candidate


def _official_target_evidence(target: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "mmm/official-target-evidence-v1",
        "retrieved_by": "mmm_authoritative_retriever",
        "authorization": "read_only_evidence",
        "sources": [
            {
                "source_url": (
                    "https://maven.fabricmc.net/net/fabricmc/fabric-api/"
                    "fabric-api/0.92.11%2B1.20.1/"
                ),
                "observed_at": "2026-08-01T00:00:00Z",
                "content_sha256": "sha256:" + "f" * 64,
                "facts": target,
            }
        ],
    }
    return _seal_technology_receipt(body, _RECEIPT_KEY)


def _technology_test_receipt(
    test_id: str,
    requirement_id: str,
    *,
    candidate_id: str,
    revision: str,
    artifact_sha256: str,
    evidence_sha256: str,
    candidate_snapshot_sha256: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "mmm/technology-test-receipt-v1",
        "executed_by": "mmm_quality_runner",
        "status": "pass",
        "test_id": test_id,
        "requirement_id": requirement_id,
        "candidate_id": candidate_id,
        "revision": revision,
        "artifact_sha256": artifact_sha256,
        "evidence_sha256": evidence_sha256,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "observed_at": "2026-08-01T00:00:00Z",
        "environment_sha256": "sha256:" + "1" * 64,
        "result_sha256": "sha256:" + "2" * 64,
    }
    return _seal_technology_receipt(body, _RECEIPT_KEY)


def test_reviewed_candidate_can_pass_but_newest_is_never_an_automatic_winner() -> None:
    radar = build_technology_radar("Use AI inference for request-derived NPC behavior.")
    requirement = _requirement(radar, "ai_inference")
    reviewed = _complete_candidate(requirement, candidate_id="older-reviewed")
    reviewed_assessment = _assess_technology_candidate_with_receipt_key(
        requirement, reviewed, receipt_key=_RECEIPT_KEY
    )
    assert reviewed_assessment["status"] == "eligible"
    assert reviewed_assessment["eligible"] is True
    assert reviewed_assessment["selection_policy"]["latest_is_automatically_best"] is False
    newest = deepcopy(reviewed)
    newest["candidate_id"] = "newest-but-unreviewed"
    newest["maintenance"] = {"last_modified": "2099-01-01T00:00:00Z"}
    newest["licenses"]["model"]["reviewed"] = False
    newest_assessment = _assess_technology_candidate_with_receipt_key(
        requirement, newest, receipt_key=_RECEIPT_KEY
    )
    assert newest_assessment["eligible"] is False
    assert "licenses" in newest_assessment["unresolved_gates"]
    assert newest_assessment["candidate"]["external_text_is_instructions"] is False


def test_official_target_receipt_cannot_be_missing_or_tampered() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    missing = _complete_candidate(requirement)
    missing["official_target_evidence"] = {}
    missing_assessment = _assess_technology_candidate_with_receipt_key(
        requirement, missing, receipt_key=_RECEIPT_KEY
    )
    assert "official_target_evidence" in missing_assessment["unresolved_gates"]
    tampered = _complete_candidate(requirement)
    tampered["official_target_evidence"]["sources"][0]["facts"]["java_version"] = "tampered-java"
    tampered_assessment = _assess_technology_candidate_with_receipt_key(
        requirement, tampered, receipt_key=_RECEIPT_KEY
    )
    assert "official_target_evidence" in tampered_assessment["blocking_gates"]


def test_public_hash_cannot_forge_code_owned_target_receipt() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    forged = _complete_candidate(requirement)
    receipt = forged["official_target_evidence"]
    assert isinstance(receipt, dict)
    receipt.pop("receipt_mac")
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(
        canonical_json(receipt).encode("utf-8")
    ).hexdigest()
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, forged, receipt_key=_RECEIPT_KEY
    )
    assert "official_target_evidence" in assessment["blocking_gates"]


def test_public_hash_cannot_forge_executed_test_receipts() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    forged = _complete_candidate(requirement)
    receipts = list(forged["tests"].values()) + [forged["fallback"]["test_receipt"]]
    for receipt in receipts:
        receipt.pop("receipt_mac")
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = "sha256:" + hashlib.sha256(
            canonical_json(receipt).encode("utf-8")
        ).hexdigest()
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, forged, receipt_key=_RECEIPT_KEY
    )
    assert "tests" in assessment["unresolved_gates"]
    assert "deterministic_fallback" in assessment["unresolved_gates"]
    assert assessment["eligible"] is False


def test_authenticated_receipts_do_not_validate_without_the_service_key() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    candidate = _complete_candidate(requirement)
    assessment = assess_technology_candidate(requirement, candidate)
    alias_assessment = assess_technology_compatibility(requirement, candidate)
    assert assessment["eligible"] is False
    assert "official_target_evidence" in assessment["unresolved_gates"]
    assert "tests" in assessment["unresolved_gates"]
    assert alias_assessment["assessment_sha256"] == assessment["assessment_sha256"]
    with pytest.raises(TypeError):
        assess_technology_compatibility(
            requirement, candidate, receipt_key=_RECEIPT_KEY
        )


def test_executed_receipts_cannot_be_replayed_for_another_candidate() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    original = _complete_candidate(requirement, candidate_id="candidate-a")
    replayed = deepcopy(original)
    replayed["candidate_id"] = "candidate-b"
    replayed["revision"] = "d" * 40
    replayed["artifact_sha256"] = "sha256:" + "e" * 64
    replayed["evidence_sha256"] = "sha256:" + "f" * 64
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, replayed, receipt_key=_RECEIPT_KEY
    )
    assert assessment["eligible"] is False
    assert "tests" in assessment["unresolved_gates"]
    assert "deterministic_fallback" in assessment["unresolved_gates"]


def test_executed_receipts_bind_the_entire_assessed_candidate_snapshot() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    changed = _complete_candidate(requirement)
    changed["benchmarks"]["p95_latency_ms"] = 79
    changed["maintenance"]["last_modified"] = "2099-01-01T00:00:00Z"
    changed["fallback"]["description"] = "A different untested fallback"
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, changed, receipt_key=_RECEIPT_KEY
    )
    assert assessment["eligible"] is False
    assert "tests" in assessment["unresolved_gates"]
    assert "deterministic_fallback" in assessment["unresolved_gates"]


def test_test_names_and_booleans_are_not_execution_evidence() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    candidate = _complete_candidate(requirement)
    candidate["tests"] = {name: True for name in requirement["required_tests"]}
    candidate["fallback"] = {
        "description": requirement["deterministic_fallback"],
        "deterministic": True,
        "tested": True,
    }
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, candidate, receipt_key=_RECEIPT_KEY
    )
    assert "tests" in assessment["unresolved_gates"]
    assert "deterministic_fallback" in assessment["unresolved_gates"]


def test_mismatched_minecraft_bridge_and_unsafe_weights_fail_closed() -> None:
    radar = build_technology_radar("Use AI inference for NPC dialogue.")
    requirement = _requirement(radar, "ai_inference")
    candidate = _complete_candidate(requirement)
    candidate["compatibility"]["minecraft_version"] = "1.21.1"
    candidate["formats"] = ["pickle", "safetensors"]
    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, candidate, receipt_key=_RECEIPT_KEY
    )
    assert assessment["status"] == "blocked"
    assert "exact_minecraft_bridge" in assessment["blocking_gates"]
    assert "safe_artifact_format" in assessment["blocking_gates"]


def test_large_research_graph_uses_bound_pages_without_a_project_wide_cap() -> None:
    domains = [
        {
            "domain_id": f"system_{index:03d}",
            "objective": f"Research requested system {index:03d}.",
            "requirements": [f"Implement system {index:03d}."],
            "evidence_kinds": ["ai_inference"],
            "queries": [f"system {index:03d} inference evidence"],
        }
        for index in range(137)
    ]
    brief = {"domains": domains}
    prompt = "Research every explicitly classified capability."
    cursor = ""
    found: list[str] = []
    pages = 0
    while True:
        page = build_technology_radar(prompt, brief, page_size=19, cursor=cursor)
        pages += 1
        found.extend(item["requirement_id"] for item in page["requirements"])
        cursor = page["pagination"]["next_cursor"]
        if not cursor:
            break
    assert pages == 8
    assert len(found) == 137
    assert len(set(found)) == 137
    assert page["pagination"]["total_requirements"] == 137
    assert "no project-wide" in page["scale_policy"]
    first = build_technology_radar(prompt, brief, page_size=19)
    with pytest.raises(SpecValidationError, match="does not match"):
        build_technology_radar(
            prompt + " changed",
            brief,
            page_size=19,
            cursor=first["pagination"]["next_cursor"],
        )
