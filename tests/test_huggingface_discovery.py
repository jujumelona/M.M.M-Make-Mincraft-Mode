import httpx
import pytest

from minecraft_mod_ai.ecosystem_discovery import (
    EcosystemDiscoveryClient,
    EcosystemDiscoveryUnavailable,
)
from minecraft_mod_ai.spec import SpecValidationError


def _client(handler) -> EcosystemDiscoveryClient:
    return EcosystemDiscoveryClient(transport=httpx.MockTransport(handler))


def test_huggingface_search_uses_card_metadata_and_bound_link_cursor() -> None:
    requests: list[httpx.Request] = []
    revision = "a" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "huggingface.co"
        assert request.url.path == "/api/models"
        assert request.url.params["search"] == "streaming speech recognition"
        assert request.url.params["limit"] == "2"
        assert request.url.params["full"] == "true"
        assert request.url.params["cardData"] == "true"
        assert request.url.params["sort"] == "lastModified"
        assert request.url.params["direction"] == "-1"
        if len(requests) == 1:
            assert "cursor" not in request.url.params
            return httpx.Response(
                200,
                headers={
                    "Link": (
                        "<https://huggingface.co/api/models?"
                        "cursor=opaque%2Btoken%3D&limit=2>; rel=\"next\""
                    )
                },
                json=[
                    {
                        "id": "example/safe-asr",
                        "sha": revision,
                        "author": "example",
                        "pipeline_tag": "automatic-speech-recognition",
                        "library_name": "transformers",
                        "lastModified": "2026-07-30T00:00:00Z",
                        "private": False,
                        "gated": False,
                        "tags": [
                            "automatic-speech-recognition",
                            "license:apache-2.0",
                        ],
                        "cardData": {
                            "license": "apache-2.0",
                            "datasets": ["example/declared-corpus"],
                            "language": ["en", "ko"],
                        },
                        "siblings": [
                            {"rfilename": "model.safetensors"},
                            {"rfilename": "README.md"},
                        ],
                    },
                    {
                        "id": "example/gated-custom-voice",
                        "sha": "b" * 40,
                        "pipeline_tag": "text-to-speech",
                        "library_name": "transformers",
                        "private": True,
                        "gated": "manual",
                        "cardData": {
                            "license": "other",
                            "license_link": "https://example.test/custom-license",
                        },
                        "siblings": [{"rfilename": "weights.bin"}],
                    },
                    {
                        "id": "example/unpinned",
                        "pipeline_tag": "text-to-speech",
                        "cardData": {"license": "mit"},
                    },
                ],
            )
        assert request.url.params["cursor"] == "opaque+token="
        return httpx.Response(200, json=[])

    client = _client(handler)
    first = client.search(
        "huggingface_models",
        "streaming speech recognition",
        target_profile="speech_ai",
        limit=2,
    )

    assert first["provider_total_estimate"] is None
    assert first["target_profile"] == "speech_ai"
    assert first["next_cursor"].startswith("token:")
    assert "opaque+token=" not in first["next_cursor"]
    assert first["download_performed"] is False
    assert len(first["candidates"]) == 2
    safe = first["candidates"][0]
    assert safe["metadata"]["revision_sha"] == revision
    assert safe["metadata"]["card"]["datasets"] == [
        "example/declared-corpus"
    ]
    assert safe["metadata"]["format_inventory"]["has_safetensors"] is True
    assert safe["reuse_status"] == "candidate_only_metadata_not_weights"
    blocked = first["candidates"][1]
    assert blocked["metadata"]["gated"] is True
    assert blocked["metadata"]["private"] is True
    assert blocked["reuse_status"] == "blocked_gated_private_or_disabled"
    assert blocked["license_policy"].startswith("reject_")

    second = client.search(
        "huggingface_models",
        "streaming speech recognition",
        target_profile="speech_ai",
        cursor=first["next_cursor"],
        limit=2,
    )
    assert second["next_cursor"] == ""
    assert len(requests) == 2

    with pytest.raises(SpecValidationError, match="target profile"):
        client.search(
            "huggingface_models",
            "streaming speech recognition",
            target_profile="ai_runtime",
            cursor=first["next_cursor"],
            limit=2,
        )
    assert len(requests) == 2


def test_github_only_injects_minecraft_terms_for_minecraft_profile() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])
        return httpx.Response(200, json={"total_count": 0, "items": []})

    client = _client(handler)
    client.search("github", "streaming inference", target_profile="minecraft_mod")
    client.search("github", "streaming inference", target_profile="speech_ai")
    client.search("github", "agent tool runtime", target_profile="ai_runtime")

    assert queries[0].startswith("streaming inference minecraft fabric ")
    assert queries[1].startswith("streaming inference in:name")
    assert queries[2].startswith("agent tool runtime in:name")
    assert "minecraft" not in queries[1]
    assert "fabric" not in queries[1]
    assert "minecraft" not in queries[2]
    assert "fabric" not in queries[2]


def test_github_provider_hard_limit_is_explicit_not_silent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"total_count": 1205, "items": []},
        )

    page = _client(handler).search(
        "github",
        "speech runtime",
        target_profile="speech_ai",
        limit=100,
    )

    assert page["provider_total_estimate"] == 1205
    assert page["provider_truncated"] is True
    assert page["provider_result_limit"] == 1000


def test_target_profile_binds_numeric_cursors_too() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"total_count": 2, "items": []})

    client = _client(handler)
    first = client.search(
        "github",
        "runtime",
        target_profile="ai_runtime",
        limit=1,
    )
    assert first["next_cursor"].startswith("page:2:")
    with pytest.raises(SpecValidationError, match="target profile"):
        client.search(
            "github",
            "runtime",
            target_profile="minecraft_mod",
            cursor=first["next_cursor"],
            limit=1,
        )
    assert len(requests) == 1


def test_huggingface_inspection_rechecks_exact_revision_without_weights() -> None:
    requests: list[httpx.Request] = []
    revision = "c" * 40
    artifact_sha = "d" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "huggingface.co"
        assert "/resolve/" not in request.url.path
        if request.url.path == "/api/models/example/voice-model":
            assert request.url.params["full"] == "true"
            assert request.url.params["cardData"] == "true"
            return httpx.Response(
                200,
                json={
                    "id": "example/voice-model",
                    "sha": revision,
                    "pipeline_tag": "text-to-speech",
                    "library_name": "transformers",
                    "private": False,
                    "gated": False,
                },
            )
        assert request.url.path == (
            f"/api/models/example/voice-model/revision/{revision}"
        )
        assert request.url.params["blobs"] == "true"
        return httpx.Response(
            200,
            json={
                "id": "example/voice-model",
                "sha": revision,
                "author": "example",
                "pipeline_tag": "text-to-speech",
                "library_name": "transformers",
                "lastModified": "2026-07-31T00:00:00Z",
                "private": False,
                "gated": False,
                "tags": ["license:apache-2.0", "text-to-speech"],
                "cardData": {
                    "license": "apache-2.0",
                    "base_model": ["example/base-model"],
                    "datasets": ["example/voice-corpus"],
                    "language": ["ko"],
                },
                "siblings": [
                    {
                        "rfilename": "model.safetensors",
                        "blobId": "e" * 40,
                        "lfs": {"sha256": artifact_sha, "size": 1234},
                    },
                    {"rfilename": "legacy/pytorch_model.bin", "size": 5678},
                    {"rfilename": "modeling_voice.py", "size": 99},
                    {"rfilename": "../escaped.bin", "size": 1},
                ],
            },
        )

    result = _client(handler).inspect_huggingface_model("example/voice-model")

    assert len(requests) == 2
    assert result["revision_sha"] == revision
    assert result["source_url"].endswith(f"/tree/{revision}")
    assert result["card"]["license_id"] == "apache-2.0"
    assert result["card"]["datasets"] == ["example/voice-corpus"]
    assert [item["path"] for item in result["files"]] == [
        "legacy/pytorch_model.bin",
        "model.safetensors",
        "modeling_voice.py",
    ]
    safe_file = next(
        item for item in result["files"] if item["path"] == "model.safetensors"
    )
    assert safe_file["lfs_sha256"] == artifact_sha
    assert safe_file["safe_data_format"] is True
    assert result["format_inventory"]["has_safetensors"] is True
    assert result["format_inventory"]["unsafe_serialization_files"] == [
        "legacy/pytorch_model.bin"
    ]
    assert result["format_inventory"]["repository_code_files"] == [
        "modeling_voice.py"
    ]
    assert result["gates"]["model_license"]["status"] == (
        "manual_review_required"
    )
    assert result["gates"]["code_license"]["status"].startswith("unresolved")
    assert result["gates"]["dataset_provenance"]["status"] == (
        "manual_review_required"
    )
    assert result["gates"]["runtime_compatibility"]["status"] == "unverified"
    assert result["selected"] is False
    assert result["download_performed"] is False
    assert result["weights_downloaded"] is False
    assert result["code_executed"] is False


def test_huggingface_inspection_rejects_revision_mismatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        revision = "a" * 40 if calls == 1 else "b" * 40
        return httpx.Response(
            200,
            json={
                "id": "example/model",
                "sha": revision,
                "cardData": {"license": "mit"},
            },
        )

    with pytest.raises(EcosystemDiscoveryUnavailable, match="pinned revision"):
        _client(handler).inspect_huggingface_model("example/model")
    assert calls == 2


def test_huggingface_metadata_api_cannot_be_used_to_download_weights() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Non-metadata paths must not reach the transport")

    client = _client(handler)
    with pytest.raises(SpecValidationError, match="metadata API paths"):
        client._get_json(
            "https://huggingface.co/example/model/resolve/main/model.safetensors"
        )
