"""New Gradle versions are checked against their exact official release pins."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import verified_scaffold_registry as scaffold


def test_live_wrapper_pin_uses_official_release_not_fabric_main(monkeypatch):
    digest = "a" * 64
    seen = []
    scaffold._live_wrapper_pin.cache_clear()

    def get(url: str, timeout: int = 30) -> str:
        seen.append(url)
        return digest + "\n"

    monkeypatch.setattr(scaffold, "_fetch_text", get)
    result = scaffold._live_wrapper_pin("9.5.1")
    assert result == (
        "https://raw.githubusercontent.com/gradle/gradle/v9.5.1/gradle/wrapper/gradle-wrapper.jar",
        digest,
        0,
    )
    assert seen == ["https://services.gradle.org/distributions/gradle-9.5.1-wrapper.jar.sha256"]
    with pytest.raises(RuntimeError, match="Invalid Gradle"):
        scaffold._live_wrapper_pin("../main")
    scaffold._live_wrapper_pin.cache_clear()


def test_requested_release_wrapper_is_checksum_verified_and_reused(tmp_path, monkeypatch):
    binary = b"release-specific-wrapper"
    digest = hashlib.sha256(binary).hexdigest()
    scaffold._live_wrapper_pin.cache_clear()
    monkeypatch.setattr(scaffold, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(scaffold, "_fetch_text", lambda url, timeout=30: digest)
    monkeypatch.setattr(scaffold, "_validate_wrapper", lambda path: None)
    calls = []

    def download(url, target):
        calls.append(url)
        target.write_bytes(binary)

    monkeypatch.setattr(scaffold, "_download", download)
    adapter = SimpleNamespace(gradle="9.5.1")
    path = scaffold._ensure_wrapper(adapter)
    assert path.read_bytes() == binary
    assert scaffold.verified_wrapper_sha256(adapter) == digest
    assert len(calls) == 1
    path.write_bytes(b"tampered")
    assert scaffold.verified_wrapper_sha256(adapter) == digest
    assert len(calls) == 2
    scaffold._live_wrapper_pin.cache_clear()


