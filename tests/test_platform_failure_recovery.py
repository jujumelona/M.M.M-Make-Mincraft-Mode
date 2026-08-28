from __future__ import annotations

import urllib.error

import pytest

from minecraft_mod_ai import platform_live_discovery as live
from minecraft_mod_ai import platform_optimizer as optimizer
from minecraft_mod_ai import reuse_planner as reuse
from minecraft_mod_ai.platform_catalog import PlatformAdapter


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_failure_recovery_test",
        edition="java",
        loader="fabric",
        minecraft_version="mmm-test-target",
        java_version="21",
        yarn_mappings="mmm-test-target+test-mappings",
        mappings_kind="yarn",
        mappings_version="mmm-test-target+test-mappings",
        fabric_loader="test-loader",
        fabric_api="test-api",
        fabric_loom="test-loom",
        gradle="test-gradle",
        gradle_sha256="0" * 64,
        data_pack_version="1",
        resource_pack_version="1",
        resource_pack_format=1,
        release_metadata_url="https://www.minecraft.net/test-fixture/mmm-test-target",
        source_api_family="fabric_reviewed_test_template",
        deterministic_module_kinds=frozenset(),
    )


def test_official_fetch_retries_transient_http_failure_and_exposes_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def open_url(request, *, timeout: int):
        del timeout
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                404,
                "edge-cache miss",
                hdrs=None,
                fp=None,
            )
        return _Response(b"ok")

    monkeypatch.setenv("MMM_PLATFORM_DISCOVERY_RETRIES", "2")
    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(live.urllib.request, "urlopen", open_url)

    assert live._fetch("https://example.test/metadata") == b"ok"
    assert calls[0] == "https://example.test/metadata"
    assert "mmm_platform_retry=2" in calls[1]
    output = capsys.readouterr().out
    assert "attempt 1/2" in output
    assert "recovered GET https://example.test/metadata on attempt 2/2" in output


def test_release_article_slug_uses_official_hyphenated_version() -> None:
    assert live._release_article_url("26.2") == (
        "https://www.minecraft.net/en-us/article/minecraft-java-edition-26-2"
    )


def test_official_fetch_exhaustion_keeps_attempt_count_and_root_cause(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def open_url(_request, *, timeout: int):
        del timeout
        raise urllib.error.URLError("proxy unavailable")

    monkeypatch.setenv("MMM_PLATFORM_DISCOVERY_RETRIES", "2")
    monkeypatch.setattr(live.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(live.urllib.request, "urlopen", open_url)

    with pytest.raises(live.PlatformDiscoveryError, match=r"after 2 attempt\(s\).*proxy unavailable"):
        live._fetch("https://example.test/metadata")
    output = capsys.readouterr().out
    assert "attempt 1/2" in output
    assert "attempt 2/2" in output
    assert "proxy unavailable" in output


def test_reuse_planner_reports_every_failed_provider_candidate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = reuse.CapabilityGraph(nodes=(), edges=(), sources=())
    monkeypatch.setattr(reuse, "decompose_capability_graph", lambda *_args, **_kwargs: graph)
    monkeypatch.setattr(
        reuse,
        "discover_target_keys",
        lambda **kwargs: (
            kwargs["diagnostics"].append("version discovery completed"),
            (("fabric", "26.2"), ("fabric", "26.1.2")),
        )[1],
    )

    def resolve(version: str, loader: str):
        raise ValueError(
            f"official platform discovery failed after 2 attempt(s): "
            f"https://www.minecraft.net/en-us/article/minecraft-java-edition-{version}: HTTP 404"
        )

    monkeypatch.setattr(reuse, "adapter_for_target", resolve)

    with pytest.raises(ValueError, match="No executable platform provider") as caught:
        reuse.optimize_platform_and_reuse("probe")

    message = str(caught.value)
    assert "26.2" in message
    assert "26.1.2" in message
    assert "HTTP 404" in message
    output = capsys.readouterr().out
    assert "target resolution skipped loader=fabric version=26.2" in output
    assert "target resolution skipped loader=fabric version=26.1.2" in output


def test_deep_evidence_failure_keeps_exact_target_as_fresh_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    adapter = _adapter()

    def fail(*_args, **_kwargs):
        raise RuntimeError("inspection backend unavailable")

    monkeypatch.setattr(optimizer, "_deep_evidence", fail)
    result = optimizer._parallel_deep(
        (adapter,),
        queries=("gameplay.core",),
        matrix={adapter.adapter_id: {}},
        client=object(),
        target_research_fn=None,
        inherited_errors=(),
        shallow_candidate_count=0,
    )

    assert len(result) == 1
    assert result[0].adapter is adapter
    assert result[0].evidence_quality == 0.0
    assert any("inspection backend unavailable" in error for error in result[0].discovery_errors)
    assert "using fresh-only evidence" in capsys.readouterr().out
