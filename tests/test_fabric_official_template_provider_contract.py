from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.fabric_official_template_provider import (
    FabricTemplateProviderError,
    _gradle_wrapper_version,
    _pin_generated_toolchain,
    _read_properties,
)


def _adapter():
    return SimpleNamespace(
        fabric_loader="0.19.5",
        fabric_api="0.136.1+1.21.8",
        fabric_loom="1.17.20",
        gradle="9.5.1",
        gradle_sha256="a" * 64,
    )


def _write_template(root: Path, *, include_api: bool = True) -> None:
    (root / "gradle/wrapper").mkdir(parents=True)
    properties = [
        "minecraft_version=1.21.8",
        "loader_version=0.19.6",
        "loom_version=1.17-SNAPSHOT",
    ]
    if include_api:
        properties.append("fabric_api_version=0.140.0+1.21.8")
    (root / "gradle.properties").write_text(
        "\n".join(properties) + "\n",
        encoding="utf-8",
    )
    (root / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "\n".join(
            (
                "distributionBase=GRADLE_USER_HOME",
                "distributionPath=wrapper/dists",
                "distributionUrl=https\\://services.gradle.org/distributions/gradle-9.7.1-bin.zip",
                "zipStoreBase=GRADLE_USER_HOME",
                "zipStorePath=wrapper/dists",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_pin_generated_toolchain_replaces_only_volatile_provider_defaults(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _write_template(root)

    _pin_generated_toolchain(root, _adapter())

    properties = _read_properties(root / "gradle.properties")
    assert properties["minecraft_version"] == "1.21.8"
    assert properties["loader_version"] == "0.19.5"
    assert properties["fabric_api_version"] == "0.136.1+1.21.8"
    assert properties["loom_version"] == "1.17.20"
    assert _gradle_wrapper_version(root) == "9.5.1"
    wrapper = (root / "gradle/wrapper/gradle-wrapper.properties").read_text(
        encoding="utf-8"
    )
    assert "distributionSha256Sum=" + ("a" * 64) in wrapper


def test_pin_generated_toolchain_fails_when_official_template_shape_drifted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _write_template(root, include_api=False)

    with pytest.raises(
        FabricTemplateProviderError,
        match="no recognized Fabric API version property",
    ):
        _pin_generated_toolchain(root, _adapter())
