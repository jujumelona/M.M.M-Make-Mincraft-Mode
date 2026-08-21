from __future__ import annotations

import os
from pathlib import Path

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_registry import ModelRegistry, _read_registry_source


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "model-registry.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_duplicate_profile_key_is_rejected_before_yaml_last_writer_wins(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: mmm/model-registry-v2
profiles:
  test:
    roles: {}
  test:
    roles: {}
""",
    )

    with pytest.raises(ModelConfigurationError, match="Duplicate model registry YAML key 'test'"):
        ModelRegistry(path)


def test_duplicate_role_key_is_rejected_before_provider_override(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: mmm/model-registry-v2
profiles:
  test:
    roles:
      planner:
        provider: local
        adapter: mock
        model_id: first
      planner:
        provider: local
        adapter: mock
        model_id: other
""",
    )

    with pytest.raises(ModelConfigurationError, match="Duplicate model registry YAML key 'planner'"):
        ModelRegistry(path)


def test_registry_parse_cache_is_bound_to_content_not_size_and_mtime(tmp_path: Path) -> None:
    first = """schema_version: mmm/model-registry-v2
profiles:
  alpha:
    roles: {}
"""
    second = """schema_version: mmm/model-registry-v2
profiles:
  bravo:
    roles: {}
"""
    assert len(first.encode("utf-8")) == len(second.encode("utf-8"))
    path = _write(tmp_path, first)
    _, profiles_one = _read_registry_source(path)
    assert set(profiles_one) == {"alpha"}

    stat = path.stat()
    path.write_text(second, encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    _, profiles_two = _read_registry_source(path)
    assert set(profiles_two) == {"bravo"}
