from __future__ import annotations

from typing import Any

from minecraft_mod_ai.mcp_tools import MMMToolService


class _DiscoveryStub:
    def __init__(self) -> None:
        self.search_call: dict[str, Any] = {}

    def search(
        self,
        provider: str,
        query: str,
        *,
        cursor: str,
        limit: int,
        target_profile: str,
    ) -> dict[str, Any]:
        self.search_call = {
            "provider": provider,
            "query": query,
            "cursor": cursor,
            "limit": limit,
            "target_profile": target_profile,
        }
        return {"schema_version": "test/page-v1", **self.search_call}

    def inspect_huggingface_model(self, repo_id: str) -> dict[str, Any]:
        return {
            "schema_version": "test/hf-inspection-v1",
            "repo_id": repo_id,
            "revision_sha": "a" * 40,
            "download_performed": False,
        }


def test_service_dispatches_profile_bound_discovery_and_hf_inspection(
    tmp_path,
) -> None:
    discovery = _DiscoveryStub()
    service = MMMToolService(
        workspace_root=tmp_path / "workspace",
        discovery_client_factory=lambda: discovery,
    )

    page = service.discover_ecosystem_resources(
        "huggingface_models",
        "streaming Korean ASR",
        "cursor-token",
        7,
        "speech_ai",
    )
    inspection = service.inspect_huggingface_model("owner/model")

    assert page["target_profile"] == "speech_ai"
    assert discovery.search_call["cursor"] == "cursor-token"
    assert inspection == {
        "schema_version": "test/hf-inspection-v1",
        "repo_id": "owner/model",
        "revision_sha": "a" * 40,
        "download_performed": False,
    }


def test_service_builds_and_fail_closed_assesses_voice_requirements(
    tmp_path,
) -> None:
    service = MMMToolService(workspace_root=tmp_path / "workspace")
    radar = service.build_technology_radar(
        "실시간 한국어 음성인식과 TTS NPC를 만들어줘."
    )
    requirement = next(
        item
        for item in radar["requirements"]
        if item["capability_kind"] == "speech_recognition"
    )

    assessment = service.assess_technology_compatibility(
        requirement,
        {
            "candidate_id": "unverified-asr",
            "capability_kind": "speech_recognition",
            "topology": "local_sidecar",
            "licenses": {},
        },
    )

    assert assessment["eligible"] is False
    assert assessment["status"] in {"blocked", "needs_evidence"}
    assert "exact_minecraft_bridge" in (
        assessment["blocking_gates"] + assessment["unresolved_gates"]
    )
    assert assessment["selection_policy"]["auto_download_or_execution"] is False
    evidence = assessment["candidate"]["official_target_evidence"]
    assert evidence["schema_version"] == "mmm/official-target-evidence-v1"
    assert evidence["receipt_mac"].startswith("hmac-sha256:")
    assert {
        source["retrieval_document_id"] for source in evidence["sources"]
    } == {
        "fabric-yarn-1201",
        "fabric-api-1201",
        "fabric-loader-01610",
        "java-17-runtime",
    }
