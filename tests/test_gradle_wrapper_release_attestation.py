"""New Gradle versions are checked against their exact official release pins."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import verified_scaffold_registry as scaffold
from minecraft_mod_ai import reuse_build_verifier as verifier


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
        "https://services.gradle.org/distributions/gradle-9.5.1-wrapper.jar",
        digest,
        0,
    )
    assert seen == [result[0] + ".sha256"]
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


def test_provider_attestation_supports_newer_official_gradle(tmp_path, monkeypatch):
    from minecraft_mod_ai import platform_catalog

    (tmp_path / "gradle" / "wrapper").mkdir(parents=True)
    wrapper = tmp_path / "gradle" / "wrapper" / "gradle-wrapper.jar"
    wrapper.write_bytes(b"new-wrapper")
    expected_sha = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    (tmp_path / "gradle" / "wrapper" / "gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/"
        "gradle-9.5.1-bin.zip\n"
        "distributionSha256Sum=" + "b" * 64 + "\n",
        encoding="utf-8",
    )
    (tmp_path / "build.gradle").write_text(
        "plugins { id 'net.fabricmc.fabric-loom-remap' version '1.0' }\n"
        "dependencies { minecraft 'com.mojang:minecraft:1.21.11' }\n"
    )
    adapter = SimpleNamespace(
        gradle="9.5.1", java_version="25", gradle_sha256="b" * 64,
    )
    monkeypatch.setattr(platform_catalog, "adapter_for_target", lambda *_: adapter)
    monkeypatch.setattr(scaffold, "validate_scaffold_buildability", lambda *_: None)
    monkeypatch.setattr(scaffold, "verified_wrapper_sha256", lambda *_: expected_sha)
    monkeypatch.setattr(verifier, "_java_major_version", lambda: "25")
    receipt = verifier._inspect_build_toolchain(tmp_path)
    assert receipt.wrapper_verified is True
    assert receipt.distribution_verified is True
    assert receipt.target_matrix_verified is True
    assert receipt.is_attested is True

    # A wrong release checksum must never gain proof merely from a valid build.
    (tmp_path / "gradle" / "wrapper" / "gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/"
        "gradle-9.5.1-bin.zip\n"
        "distributionSha256Sum=" + "c" * 64 + "\n",
        encoding="utf-8",
    )
    receipt = verifier._inspect_build_toolchain(tmp_path)
    assert receipt.is_attested is False
