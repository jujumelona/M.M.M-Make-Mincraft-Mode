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
    compute_voice_language_intersection,
    technology_research_routes,
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
            "real_time_factor": 0.7,
        },
        "privacy": {
            "raw_input_leaves_device": False,
            "retention_policy": "none",
            "deletion_supported": True,
        },
        "voice_rights": {
            "explicit_consent": True,
            "authorized_speaker": True,
            "provenance_verified": True,
            "provenance": "signed owner recording receipt",
            "revocation_supported": True,
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


def test_unrelated_game_request_does_not_inject_ai_or_voice() -> None:
    radar = build_technology_radar(
        "Create a farming loop with planting, watering and a market."
    )

    assert radar["requirements"] == []
    assert radar["classification"] == {
        "ai_requested": False,
        "agent_tools_requested": False,
        "voice_requested": False,
        "voice_adaptation_requested": False,
        "translation_requested": False,
        "offline_required": False,
        "real_time_requested": False,
    }
    assert radar["voice_contract"]["activated"] is False


def test_audio_assets_are_not_misclassified_as_a_voice_pipeline() -> None:
    radar = build_technology_radar(
        "Add ambient sound effects, custom OGG music and subtitles."
    )

    assert radar["classification"]["voice_requested"] is False
    assert radar["requirements"] == []


def test_voice_request_is_decomposed_and_keeps_identity_separate_from_expression() -> None:
    radar = build_technology_radar(
        "실시간 음성인식, TTS, 내 목소리 LoRA를 쓰는 AI NPC를 만들어줘."
    )
    kinds = {item["capability_kind"] for item in radar["requirements"]}

    assert {
        "speech_recognition",
        "voice_activity_detection",
        "speech_synthesis",
        "voice_adaptation",
        "voice_transport",
        "language_intersection",
        "ai_inference",
    } <= kinds
    assert radar["classification"]["voice_adaptation_requested"] is True
    voice = radar["voice_contract"]
    assert voice["speaker_identity"] == "speech_synthesis_or_voice_model"
    assert voice["expression"] == {
        "owner": "utterance_local_pattern_trace",
        "representation": "time_series",
        "fields": ["time", "energy", "entropy", "f0", "attack", "pause"],
        "prohibited": ["single_embedding", "conversation_average"],
    }
    assert voice["adaptation"]["status"].startswith("blocked_until_consent")


def test_korean_possessive_voice_adaptation_is_not_missed() -> None:
    radar = build_technology_radar(
        "사용자가 소유하며 명시적으로 동의한 목소리는 선택적으로 적응한다."
    )

    assert radar["classification"]["voice_adaptation_requested"] is True
    assert _requirement(radar, "voice_adaptation")


def test_realtime_voice_requires_capture_cancellation_and_transport_stress_tests() -> None:
    radar = build_technology_radar(
        "실시간 마이크 음성인식과 TTS로 대화하는 NPC를 만들어줘."
    )

    asr_tests = set(_requirement(radar, "speech_recognition")["required_tests"])
    tts_tests = set(_requirement(radar, "speech_synthesis")["required_tests"])
    transport_tests = set(
        _requirement(radar, "voice_transport")["required_tests"]
    )
    assert {
        "microphone_permission_and_capture",
        "echo_noise_and_silence_conditions",
        "barge_in_and_cancellation",
    } <= asr_tests
    assert {
        "streaming_playback_backpressure",
        "barge_in_and_cancellation",
    } <= tts_tests
    assert {
        "jitter_packet_loss_and_backpressure",
        "barge_in_and_cancellation",
    } <= transport_tests
    assert radar["target_evidence_policy"] == {
        "coordinates_are_declared_constraints": True,
        "official_exact_version_receipt_required": True,
        "current_documentation_requires_target_translation": True,
        "receipt_schema": "mmm/official-target-evidence-v1",
        "authenticated_code_owned_mac_required": True,
        "executed_tests_bind_candidate_snapshot": True,
    }


def test_target_and_authority_contract_are_exact_for_every_requirement() -> None:
    radar = build_technology_radar("Add speech recognition to NPC dialogue.")

    for requirement in radar["requirements"]:
        assert requirement["target"] == {
            "edition": "java",
            "minecraft_version": "1.20.1",
            "loader": "fabric",
            "mappings": "yarn-1.20.1+build.1",
            "java_version": "17",
            "fabric_loader": "0.16.10",
            "fabric_api": "0.92.11+1.20.1",
        }
        assert requirement["authority"]["game_state_mutation"] == "server_only"
        assert (
            requirement["authority"]["client_messages"]
            == "schema_validated_and_rate_limited_by_server"
        )

    with pytest.raises(SpecValidationError, match="exact Minecraft 1.20.1"):
        build_technology_radar(
            "AI NPC",
            target={"minecraft_version": "1.21.1"},
        )


def test_offline_request_removes_remote_api_without_removing_other_options() -> None:
    radar = build_technology_radar(
        "인터넷 없이 로컬만 사용하는 음성 인식과 TTS를 만들어줘."
    )

    for requirement in radar["requirements"]:
        assert "remote_api" not in requirement["allowed_topologies"]
    synthesis = _requirement(radar, "speech_synthesis")
    assert "in_process_java" in synthesis["allowed_topologies"]
    assert "local_sidecar" in synthesis["allowed_topologies"]
    assert "offline_build_tool" in synthesis["allowed_topologies"]


def test_voice_adaptation_is_blocked_without_consent_and_lifecycle_controls() -> None:
    radar = build_technology_radar("Adapt my voice with a consented voice LoRA.")
    requirement = _requirement(radar, "voice_adaptation")
    candidate = _complete_candidate(requirement)
    candidate["voice_rights"] = {}

    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, candidate, receipt_key=_RECEIPT_KEY
    )

    assert assessment["status"] == "blocked"
    assert assessment["eligible"] is False
    assert "voice_rights" in assessment["blocking_gates"]
    voice_gate = next(
        gate for gate in assessment["gates"] if gate["gate_id"] == "voice_rights"
    )
    assert "explicit_consent" in voice_gate["reason"]
    assert "revocation_supported" in voice_gate["reason"]
    assert "deletion_supported" in voice_gate["reason"]


def test_reviewed_candidate_can_pass_but_newest_is_never_an_automatic_winner() -> None:
    radar = build_technology_radar("Use AI inference for request-derived NPC behavior.")
    requirement = _requirement(radar, "ai_inference")
    reviewed = _complete_candidate(requirement, candidate_id="older-reviewed")

    reviewed_assessment = _assess_technology_candidate_with_receipt_key(
        requirement, reviewed, receipt_key=_RECEIPT_KEY
    )

    assert reviewed_assessment["status"] == "eligible"
    assert reviewed_assessment["eligible"] is True
    assert (
        reviewed_assessment["selection_policy"]["latest_is_automatically_best"]
        is False
    )

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
    tampered["official_target_evidence"]["sources"][0]["facts"][
        "java_version"
    ] = "21"
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
    receipts = list(forged["tests"].values()) + [
        forged["fallback"]["test_receipt"]
    ]
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
            requirement,
            candidate,
            receipt_key=_RECEIPT_KEY,
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
    candidate["tests"] = {
        name: True for name in requirement["required_tests"]
    }
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


def test_remote_voice_data_requires_transfer_consent_retention_and_deletion() -> None:
    radar = build_technology_radar("Use speech recognition for NPC dialogue.")
    requirement = _requirement(radar, "speech_recognition")
    candidate = _complete_candidate(requirement)
    candidate["topology"] = "remote_api"
    candidate["formats"] = []
    candidate["artifact_sha256"] = ""
    candidate["privacy"] = {
        "raw_input_leaves_device": True,
        "explicit_transfer_consent": False,
        "retention_policy": "",
        "deletion_supported": False,
    }

    assessment = _assess_technology_candidate_with_receipt_key(
        requirement, candidate, receipt_key=_RECEIPT_KEY
    )

    assert assessment["status"] == "blocked"
    assert "privacy" in assessment["blocking_gates"]
    format_gate = next(
        gate
        for gate in assessment["gates"]
        if gate["gate_id"] == "safe_artifact_format"
    )
    assert format_gate["status"] == "not_applicable"


def test_language_support_is_the_full_pipeline_intersection_not_a_union() -> None:
    direct = compute_voice_language_intersection(
        ["ko", "en", "ja"],
        ["en", "ko", "de"],
    )
    translated = compute_voice_language_intersection(
        ["ko", "en", "ja"],
        ["en", "ko", "de"],
        [("ko", "en"), ("ja", "ko"), ("en", "fr")],
    )

    assert direct["direct_languages"] == ["en", "ko"]
    assert direct["advertise_component_union"] is False
    assert translated["direct_languages"] == []
    assert translated["full_pipeline_paths"] == [
        {"input": "ko", "output": "en"},
        {"input": "ja", "output": "ko"},
    ]


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
        page = build_technology_radar(
            prompt,
            brief,
            page_size=19,
            cursor=cursor,
        )
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


def test_routes_are_request_derived_read_only_and_include_model_and_runtime_evidence() -> None:
    radar = build_technology_radar("Add generic speech recognition.", page_size=2)
    routes = technology_research_routes(radar)
    providers = {route["provider"] for route in routes}

    assert {"official_docs", "huggingface_models", "github", "runtime"} <= providers
    assert all(route["authorization"] == "read_only_evidence" for route in routes)
    assert all(route["query_sha256"].startswith("sha256:") for route in routes)
    assert radar["discovery_policy"]["download_or_execution_authorized"] is False
    assert "no embedded product or model list" in radar["discovery_policy"]["catalog"]
