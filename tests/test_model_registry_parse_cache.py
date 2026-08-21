from __future__ import annotations

from minecraft_mod_ai import model_registry


def test_registry_yaml_parse_is_reused_until_file_content_changes(monkeypatch, tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "schema_version: mmm/model-registry-v2\nprofiles:\n  one: {}\n",
        encoding="utf-8",
    )
    model_registry._REGISTRY_SOURCE_CACHE.clear()
    calls = 0
    original = model_registry.safe_load_unique_keys

    def counted(value, *, source):
        nonlocal calls
        calls += 1
        return original(value, source=source)

    monkeypatch.setattr(model_registry, "safe_load_unique_keys", counted)
    first = model_registry._read_registry_source(path)
    second = model_registry._read_registry_source(path)
    assert first == second
    assert calls == 1

    path.write_text(
        "schema_version: mmm/model-registry-v2\nprofiles:\n  two: {}\n",
        encoding="utf-8",
    )
    third = model_registry._read_registry_source(path)
    assert "two" in third[1]
    assert calls == 2
