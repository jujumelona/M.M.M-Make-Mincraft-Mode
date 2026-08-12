from __future__ import annotations

import os

from minecraft_mod_ai import model_registry


def test_registry_yaml_parse_is_reused_until_file_changes(monkeypatch, tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "schema_version: mmm/model-registry-v2\nprofiles:\n  one: {}\n",
        encoding="utf-8",
    )
    model_registry._REGISTRY_SOURCE_CACHE.clear()
    calls = 0
    original = model_registry.yaml.safe_load

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(model_registry.yaml, "safe_load", counted)
    first = model_registry._read_registry_source(path)
    second = model_registry._read_registry_source(path)
    assert first == second
    assert calls == 1

    path.write_text(
        "schema_version: mmm/model-registry-v2\nprofiles:\n  two: {}\n",
        encoding="utf-8",
    )
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    third = model_registry._read_registry_source(path)
    assert "two" in third[1]
    assert calls == 2
