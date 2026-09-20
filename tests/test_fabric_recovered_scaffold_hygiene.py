import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import fabric_immutable_rebind_contract as rebind
from minecraft_mod_ai import fabric_official_template_provider as provider
from minecraft_mod_ai import platform_live_execution_contract as live


def _install_recovery(monkeypatch, *, safety_failure=False):
    def interrupted_bootstrap(*, project_root, **kwargs):
        root = Path(project_root)
        resources = root / "src/main/resources"
        resources.mkdir(parents=True)
        (root / "gradle.properties").write_text("minecraft_version=26.2\n", encoding="utf-8")
        (resources / "fabric.mod.json").write_text(json.dumps({
            "id": "demo", "description": "example", "entrypoints": {"main": ["demo.Example"]},
        }), encoding="utf-8")
        source = root / "src/main/java/demo/Example.java"
        source.parent.mkdir(parents=True)
        source.write_text("// demo", encoding="utf-8")
        (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
        if safety_failure:
            raise provider.FabricTemplateSafetyError("unsafe template")
        raise provider.FabricTemplateProviderError("Java target changed after discovery")

    monkeypatch.setattr(provider, "bootstrap_fabric_project", interrupted_bootstrap)
    monkeypatch.setattr(live, "bootstrap_fabric_project", live.bootstrap_fabric_project)
    monkeypatch.setattr(live, "_write_approved_target_lock", live._write_approved_target_lock)
    monkeypatch.setattr(rebind, "_INSTALLED", False)
    monkeypatch.setattr(rebind, "_rebind_scaffold", lambda *args: {})
    monkeypatch.setattr(rebind, "_write_full_platform_lock", lambda *args: None)
    monkeypatch.setattr(provider, "_gradle_wrapper_version", lambda *args: "9.5.1")
    monkeypatch.setattr(provider, "_java_release", lambda *args: "25")
    monkeypatch.setattr(provider, "_ensure_deno", lambda *args: Path("deno"))
    monkeypatch.setattr(provider, "_deno_version", lambda *args: "test")
    rebind.install()
    return SimpleNamespace(minecraft_version="26.2", java_version="25", loader="fabric",
        fabric_loader="loader", fabric_api="api", fabric_loom="loom", gradle="9.5.1",
        mappings_kind="mojang")


def test_recovered_fresh_scaffold_gets_cleanup_before_host_contracts(tmp_path, monkeypatch):
    adapter = _install_recovery(monkeypatch)
    root = tmp_path / "new"
    spec = SimpleNamespace(mod_id="demo", package_name="demo", mod_name="Demo", summary="Real request")
    receipt = provider.bootstrap_fabric_project(project_root=root, spec=spec,
        adapter=adapter, cache_root=tmp_path / "cache")
    metadata = json.loads((root / "src/main/resources/fabric.mod.json").read_text(encoding="utf-8"))
    assert receipt["approval_rebind"] == "EXACT"
    assert metadata["description"] == "Real request"
    assert metadata["entrypoints"]["main"] == ["demo.DemoMod"]
    assert not (root / "src/main/java/demo/Example.java").exists()


def test_recovery_does_not_rewrite_existing_project(tmp_path, monkeypatch):
    adapter = _install_recovery(monkeypatch)
    existing = tmp_path / "owned.java"
    existing.write_text("user source", encoding="utf-8")
    with pytest.raises(provider.FabricTemplateProviderError, match="empty"):
        provider.bootstrap_fabric_project(project_root=tmp_path, spec=None,
            adapter=adapter, cache_root=tmp_path / "cache")
    assert existing.read_text(encoding="utf-8") == "user source"


def test_recovery_does_not_swallow_template_safety_failure(tmp_path, monkeypatch):
    adapter = _install_recovery(monkeypatch, safety_failure=True)
    with pytest.raises(provider.FabricTemplateSafetyError, match="unsafe"):
        provider.bootstrap_fabric_project(project_root=tmp_path / "new", spec=None,
            adapter=adapter, cache_root=tmp_path / "cache")
