from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.fabric_official_template_provider import (
    FabricTemplateProviderError,
    _gradle_wrapper_version,
    _install_host_runtime_contract,
    _pin_generated_toolchain,
    _read_properties,
)


def test_fresh_scaffold_removes_demo_code_and_metadata_before_host_install(tmp_path):
    import json

    from minecraft_mod_ai import fabric_official_template_provider as provider

    root = tmp_path / "project"
    resources = root / "src/main/resources"
    resources.mkdir(parents=True)
    metadata = resources / "fabric.mod.json"
    metadata.write_text(json.dumps({
        "id": "demo", "name": "Demo", "description": "This is an example description!",
        "authors": ["Me!"], "contact": {"sources": "https://github.com/FabricMC/fabric-example-mod"},
        "license": "CC0-1.0", "entrypoints": {"main": ["example.TemplateMain"]},
        "mixins": ["demo.mixins.json", {"config": "demo.client.mixins.json", "environment": "client"}],
    }), encoding="utf-8")
    for relative in ("src/main/java/example/TemplateMain.java",
                     "src/main/java/example/mixin/ExampleMixin.java"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// official example", encoding="utf-8")
    (resources / "demo.mixins.json").write_text("{}", encoding="utf-8")
    client_resources = root / "src/client/resources"
    client_resources.mkdir(parents=True)
    (client_resources / "demo.client.mixins.json").write_text("{}", encoding="utf-8")
    (root / "LICENSE").write_text("CC0 template license", encoding="utf-8")
    spec = SimpleNamespace(mod_id="demo", mod_name="Demo", package_name="example",
                           summary="A cooking mod with seasonal crops.")
    provider._clean_fresh_template_examples(root, spec)
    provider._install_host_runtime_contract(root, spec, _adapter())
    result = json.loads(metadata.read_text(encoding="utf-8"))
    assert result["description"] == spec.summary
    assert result["authors"] == []
    assert result["contact"] == {}
    assert result["entrypoints"]["main"] == ["example.DemoMod"]
    assert not result.get("mixins")
    assert not (resources / "demo.mixins.json").exists()
    assert not (client_resources / "demo.client.mixins.json").exists()
    assert not (root / "src/main/java/example/TemplateMain.java").exists()
    assert not (root / "src/main/java/example/mixin/ExampleMixin.java").exists()
    assert (root / "LICENSE").read_text(encoding="utf-8") == "CC0 template license"


def _adapter():
    return SimpleNamespace(
        minecraft_version="1.21.8",
        java_version="21",
        fabric_loader="0.19.5",
        fabric_api="0.136.1+1.21.8",
        fabric_loom="1.17.20",
        gradle="9.5.1",
        gradle_sha256="a" * 64,
    )


@pytest.mark.parametrize("linked", ["resources", "client_resources", "README.md"])
def test_scaffold_cleanup_never_mutates_symlinked_external_files(tmp_path, linked):
    import json

    from minecraft_mod_ai import fabric_official_template_provider as provider

    root = tmp_path / "project"
    resources = root / "src/main/resources"
    resources.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = json.dumps({"mixins": ["demo.mixins.json"]})
    (outside / "fabric.mod.json").write_text(original, encoding="utf-8")
    (outside / "demo.mixins.json").write_text("external mixin", encoding="utf-8")
    (outside / "README.md").write_text("external readme", encoding="utf-8")
    (resources / "fabric.mod.json").write_text(original, encoding="utf-8")
    try:
        if linked == "resources":
            (resources / "fabric.mod.json").unlink()
            resources.rmdir()
            resources.symlink_to(outside, target_is_directory=True)
        elif linked == "client_resources":
            client = root / "src/client/resources"
            client.parent.mkdir()
            client.symlink_to(outside, target_is_directory=True)
        else:
            (root / linked).symlink_to(outside / linked)
    except OSError:
        pytest.skip("host does not permit symlinks")
    spec = SimpleNamespace(summary="request", mod_name="Demo")
    with pytest.raises(provider.FabricTemplateSafetyError):
        provider._clean_fresh_template_examples(root, spec)
    assert (outside / "fabric.mod.json").read_text(encoding="utf-8") == original
    assert (outside / "demo.mixins.json").read_text(encoding="utf-8") == "external mixin"
    assert (outside / "README.md").read_text(encoding="utf-8") == "external readme"


@pytest.mark.parametrize("relative", ["src/main/resources/fabric.mod.json", "README.md"])
def test_cleanup_preflights_resolved_destinations_before_deleting_sources(tmp_path, monkeypatch, relative):
    from minecraft_mod_ai import fabric_official_template_provider as provider

    root = tmp_path / "project"
    resources = root / "src/main/resources"
    resources.mkdir(parents=True)
    (resources / "fabric.mod.json").write_text("{}", encoding="utf-8")
    source = root / "src/Example.java"
    source.write_text("original scaffold", encoding="utf-8")
    original_resolve = Path.resolve

    def redirected(path, *args, **kwargs):
        if path == root / relative:
            return tmp_path / "external" / path.name
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(provider.FabricTemplateSafetyError):
        provider._clean_fresh_template_examples(root, SimpleNamespace(summary="request", mod_name="Demo"))
    assert source.read_text(encoding="utf-8") == "original scaffold"


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
        "distributionBase=GRADLE_USER_HOME\n"
        "distributionPath=wrapper/dists\n"
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-9.7.1-bin.zip\n"
        "zipStoreBase=GRADLE_USER_HOME\n"
        "zipStorePath=wrapper/dists\n",
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


def test_host_runtime_contract_normalizes_metadata_main_and_lang(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src/main/resources").mkdir(parents=True)
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/main/resources/fabric.mod.json").write_text(
        """{
  "schemaVersion": 1,
  "id": "demo_mod",
  "version": "1.0.0",
  "environment": "*",
  "entrypoints": {
    "main": ["example.TemplateMain"]
  },
  "depends": {
    "fabricloader": ">=0.16.0",
    "minecraft": "~1.21.8",
    "java": ">=21",
    "fabric-api": "*"
  }
}
""",
        encoding="utf-8",
    )
    (root / "src/main/java/example/TemplateMain.java").write_text(
        "package example; public final class TemplateMain {}\n",
        encoding="utf-8",
    )
    spec = SimpleNamespace(
        mod_id="demo_mod",
        package_name="example",
    )

    receipt = _install_host_runtime_contract(root, spec, _adapter())

    metadata = __import__("json").loads(
        (root / "src/main/resources/fabric.mod.json").read_text(encoding="utf-8")
    )
    assert metadata["depends"] == {
        "fabricloader": "0.19.5",
        "minecraft": "1.21.8",
        "java": "21",
        "fabric-api": "0.136.1+1.21.8",
    }
    assert "example.DemoModMod" in metadata["entrypoints"]["main"]
    main = root / "src/main/java/example/DemoModMod.java"
    assert main.is_file()
    assert "implements ModInitializer" in main.read_text(encoding="utf-8")
    for locale in ("en_us", "ko_kr"):
        lang = root / f"src/main/resources/assets/demo_mod/lang/{locale}.json"
        assert lang.read_text(encoding="utf-8") == "{}\n"
    assert receipt["main_entrypoint"] == "example.DemoModMod"
    assert receipt["depends"] == metadata["depends"]


def test_host_runtime_contract_preserves_existing_language_content(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "src/main/resources/assets/demo_mod/lang").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True, exist_ok=True)
    (root / "src/main/resources/fabric.mod.json").write_text(
        '{"id":"demo_mod","entrypoints":{"main":[]},"depends":{}}\n',
        encoding="utf-8",
    )
    existing = root / "src/main/resources/assets/demo_mod/lang/en_us.json"
    existing.write_text('{"item.demo_mod.token":"Token"}\n', encoding="utf-8")
    spec = SimpleNamespace(mod_id="demo_mod", package_name="example")

    _install_host_runtime_contract(root, spec, _adapter())

    assert existing.read_text(encoding="utf-8") == (
        '{"item.demo_mod.token":"Token"}\n'
    )
