import json
import sys
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import fabric_official_template_provider as provider


def test_cached_official_cli_runs_with_compatible_node_without_reinstall(monkeypatch, tmp_path):
    node = tmp_path / "node"
    cache = tmp_path / "cache"
    package = cache / "fabric-cli" / "node_modules" / "@fabricmc" / "cli"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"version": "0.1.482"}))
    (package / "bundled.mjs").write_text("console.log('cached official CLI')")
    monkeypatch.setattr(provider, "_ensure_node", lambda _cache: node)
    monkeypatch.setattr(provider, "_runtime_version", lambda _: "v22.23.3")

    command, receipt = provider._ensure_fabric_cli(cache)

    assert command == [str(node), str(package / "bundled.mjs")]
    assert receipt["package"] == "@fabricmc/cli"
    assert receipt["version"] == "0.1.482"
    assert int(receipt["node"].lstrip("v").split(".")[0]) >= 22


def test_npm_cli_failure_reports_before_scaffold_processing(monkeypatch, tmp_path):
    script = tmp_path / "cli.py"
    script.write_text("import sys; print('official CLI unavailable', file=sys.stderr); sys.exit(3)")
    monkeypatch.setattr(provider, "_ensure_fabric_cli", lambda _cache: (
        [sys.executable, str(script)], {"package": "@fabricmc/cli", "version": "0.1.482", "node": "v22"},
    ))
    with pytest.raises(provider.FabricTemplateProviderError, match="official CLI unavailable"):
        provider.bootstrap_fabric_project(
            project_root=tmp_path / "project",
            spec=SimpleNamespace(mod_id="demo", mod_name="Demo", package_name="demo"),
            adapter=SimpleNamespace(minecraft_version="26.2"),
            cache_root=tmp_path / "cache",
        )
    assert not (tmp_path / "project").exists()


def test_cli_timeout_has_provider_error_and_no_unbounded_process(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(provider, "_ensure_fabric_cli", lambda _cache: (
        ["node", "fabric.mjs"], {},
    ))

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 240
        assert kwargs["stdin"] is subprocess.DEVNULL
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(provider.subprocess, "run", timed_out)
    with pytest.raises(provider.FabricTemplateProviderError, match="timed out"):
        provider.bootstrap_fabric_project(
            project_root=tmp_path / "project",
            spec=SimpleNamespace(mod_id="demo", mod_name="Demo", package_name="demo"),
            adapter=SimpleNamespace(minecraft_version="26.2"), cache_root=tmp_path / "cache",
        )
