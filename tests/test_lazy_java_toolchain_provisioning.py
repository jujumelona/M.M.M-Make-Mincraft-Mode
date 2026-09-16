from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import java_lsp, jdtls_bootstrap


def test_java_lsp_resolver_lazy_provisions_missing_major(monkeypatch, tmp_path: Path) -> None:
    java21 = tmp_path / "java21"
    java21.mkdir()
    java25 = tmp_path / "java25"
    java25.mkdir()

    monkeypatch.setattr(java_lsp, "_candidate_java_homes", lambda required: [java21])
    monkeypatch.setattr(java_lsp, "_java_major_versions", lambda homes: [21])
    calls: list[int | None] = []

    def provision(required_major: int | None = None) -> Path:
        calls.append(required_major)
        return java25

    monkeypatch.setattr(jdtls_bootstrap, "ensure_project_jdk", provision)

    assert java_lsp._resolve_project_java_home(25) == java25.resolve()
    assert calls == [25]


def test_ensure_project_jdk_sets_shared_project_home(monkeypatch, tmp_path: Path) -> None:
    java25 = tmp_path / "java25"
    (java25 / "bin").mkdir(parents=True)
    java = java25 / "bin" / "java"
    java.write_text("", encoding="utf-8")

    monkeypatch.setattr(jdtls_bootstrap, "_find_matching_jdk", lambda major: None)
    monkeypatch.setattr(jdtls_bootstrap, "_install_project_jdk", lambda major: java25)
    monkeypatch.setattr(jdtls_bootstrap, "_java_major", lambda executable: 25)
    monkeypatch.delenv("MMM_PROJECT_JAVA_HOME", raising=False)

    resolved = jdtls_bootstrap.ensure_project_jdk(25)

    assert resolved == java25.resolve()
    assert jdtls_bootstrap.os.environ["MMM_PROJECT_JAVA_HOME"] == str(java25.resolve())


def test_project_jdk_download_uses_adoptium_binary_endpoint(monkeypatch) -> None:
    checksum = "a" * 64
    metadata = [
        {
            "binary": {
                "package": {
                    "link": "https://example.invalid/direct-release-asset.tar.gz",
                    "checksum": checksum,
                }
            }
        }
    ]
    monkeypatch.setattr(jdtls_bootstrap.sys, "platform", "linux")
    monkeypatch.setattr(jdtls_bootstrap, "_adoptium_architecture", lambda: "x64")
    monkeypatch.setattr(
        jdtls_bootstrap,
        "_read_json",
        lambda url, max_bytes: metadata,
    )

    url, observed_checksum = jdtls_bootstrap._project_jdk_package(25)

    assert url == (
        "https://api.adoptium.net/v3/binary/latest/25/ga/linux/x64/"
        "jdk/hotspot/normal/eclipse"
    )
    assert observed_checksum == checksum
