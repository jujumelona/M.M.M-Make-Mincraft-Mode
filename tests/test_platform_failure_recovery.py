from __future__ import annotations

import urllib.error

import pytest

from minecraft_mod_ai import platform_live_discovery as live
from minecraft_mod_ai import platform_selection_pipeline as strict_platform


def test_official_fetch_single_attempt_preserves_root_cause_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def open_url(request, *, timeout: int):
        del timeout
        calls.append(request.full_url)
        raise urllib.error.URLError("proxy unavailable")

    monkeypatch.setattr(live.urllib.request, "urlopen", open_url)

    with pytest.raises(live.PlatformDiscoveryError, match="proxy unavailable"):
        live._fetch("https://example.test/metadata", retries=1)
    assert calls == ["https://example.test/metadata"]


def test_release_article_slug_uses_official_hyphenated_version() -> None:
    assert live._release_article_url("26.2") == (
        "https://www.minecraft.net/en-us/article/minecraft-java-edition-26-2"
    )


def test_canonical_platform_selector_does_not_call_legacy_discover_then_reresolve() -> None:
    source = open(strict_platform.__file__, encoding="utf-8").read()
    assert "discover_target_keys" not in source
    assert "adapter_for_target" not in source
    assert "target resolution skipped" not in source


def test_canonical_deep_stage_has_no_fresh_only_recovery_text() -> None:
    source = open(strict_platform.__file__, encoding="utf-8").read()
    assert "using fresh-only evidence" not in source
