from __future__ import annotations

from minecraft_mod_ai import reuse_build_verifier as verifier


def test_build_toolchain_target_matrix_uses_executable_provider(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(verifier, "_java_major_version", lambda: "21")
    wrapper_dir = tmp_path / "gradle" / "wrapper"
    wrapper_dir.mkdir(parents=True)
    (wrapper_dir / "gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-8.10.2-bin.zip\n"
        + "distributionSha256Sum="
        + ("0" * 64)
        + "\n",
        encoding="utf-8",
    )
    (wrapper_dir / "gradle-wrapper.jar").write_bytes(b"synthetic-test-wrapper")
    (tmp_path / "build.gradle").write_text(
        "plugins { id 'fabric-loom' version 'test-loom' }\n"
        "dependencies { minecraft 'com.mojang:minecraft:1.21.1' }\n",
        encoding="utf-8",
    )

    receipt = verifier._inspect_build_toolchain(tmp_path)

    assert receipt.loader == "fabric"
    assert receipt.minecraft_version == "1.21.1"
    assert receipt.gradle_version == "8.10.2"
    assert receipt.java_version == "21"
    assert receipt.target_matrix_verified is True
