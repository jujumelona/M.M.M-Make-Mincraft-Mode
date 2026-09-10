from __future__ import annotations

from threading import Barrier
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import generation_concurrency_safety as safety
from minecraft_mod_ai import project_index as project_index_module
from minecraft_mod_ai.project_index import ProjectIndex


def test_infers_nested_reviewed_output_paths():
    config = {
        "output_path": "src/main/resources/data/demo/a.json",
        "nested": {
            "files": [
                "src/main/java/demo/A.java",
                "src/main/java/demo/B.java",
            ]
        },
    }
    assert safety._inferred_config_anchors(config) == (
        "file://src/main/java/demo/A.java",
        "file://src/main/java/demo/B.java",
        "file://src/main/resources/data/demo/a.json",
    )


def test_rejects_unsafe_or_ambiguous_paths():
    config = {
        "path": "../escape.txt",
        "target_path": "/absolute.txt",
        "files": ["safe/out.txt", "a/../b.txt"],
    }
    assert safety._inferred_config_anchors(config) == ("file://safe/out.txt",)


def test_builtin_generators_keep_only_required_shared_collision_domains():
    content = SimpleNamespace(kind="item", module_id="item_a")
    integration = SimpleNamespace(kind="integration")
    entity = SimpleNamespace(kind="entity")
    quest = SimpleNamespace(kind="quest")
    economy = SimpleNamespace(kind="economy")
    skill = SimpleNamespace(kind="skill")
    klass = SimpleNamespace(kind="class")

    assert safety._builtin_shared_anchors(content, "content")
    assert safety._builtin_shared_anchors(integration, "content") == ()
    assert safety._builtin_shared_anchors(entity, "entity") == ()

    quest_anchor = safety._builtin_shared_anchors(quest, "system")
    economy_anchor = safety._builtin_shared_anchors(economy, "system")
    assert quest_anchor
    assert economy_anchor
    assert quest_anchor != economy_anchor
    assert safety._builtin_shared_anchors(skill, "system") == safety._builtin_shared_anchors(
        klass, "system"
    )


def test_entity_pipeline_defaults_to_singleton_nodes(monkeypatch):
    monkeypatch.delenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", raising=False)
    safety._configure_pipeline_granularity()
    assert safety.os.environ["MMM_ENTITY_PIPELINE_SHARD_SIZE"] == "1"


def test_entity_pipeline_preserves_explicit_batching(monkeypatch):
    monkeypatch.setenv("MMM_ENTITY_PIPELINE_SHARD_SIZE", "3")
    safety._configure_pipeline_granularity()
    assert safety.os.environ["MMM_ENTITY_PIPELINE_SHARD_SIZE"] == "3"


def test_system_pipeline_defaults_to_java_shard_budget(monkeypatch):
    monkeypatch.delenv("MMM_SYSTEM_PIPELINE_SHARD_SIZE", raising=False)
    monkeypatch.setenv("MMM_JAVA_SHARD_SIZE", "7")
    safety._configure_pipeline_granularity()
    assert safety.os.environ["MMM_SYSTEM_PIPELINE_SHARD_SIZE"] == "7"


def test_system_pipeline_preserves_explicit_batching(monkeypatch):
    monkeypatch.setenv("MMM_JAVA_SHARD_SIZE", "48")
    monkeypatch.setenv("MMM_SYSTEM_PIPELINE_SHARD_SIZE", "5")
    safety._configure_pipeline_granularity()
    assert safety.os.environ["MMM_SYSTEM_PIPELINE_SHARD_SIZE"] == "5"


def test_cpu_generation_width_accepts_explicit_host_capacity(monkeypatch):
    scheduler = SimpleNamespace(_cpu_capacity=lambda: 4)
    safety._install_cpu_capacity_policy(scheduler)
    monkeypatch.setenv("MMM_CPU_IO_WORKERS", "12")
    assert scheduler._cpu_capacity() == 12


def test_cpu_generation_width_fails_closed_on_invalid_value(monkeypatch):
    scheduler = SimpleNamespace(_cpu_capacity=lambda: 4)
    safety._install_cpu_capacity_policy(scheduler)
    monkeypatch.setenv("MMM_CPU_IO_WORKERS", "0")
    with pytest.raises(ValueError, match="MMM_CPU_IO_WORKERS"):
        scheduler._cpu_capacity()


def test_project_index_initial_file_work_overlaps_and_preserves_order(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_PROJECT_INDEX_WORKERS", "2")
    safety._install_project_index_parallel_scan(project_index_module)
    first = tmp_path / "a.java"
    second = tmp_path / "b.json"
    first.write_text("class A {}", encoding="utf-8")
    second.write_text('{"b": true}', encoding="utf-8")

    entered = Barrier(2)
    original = ProjectIndex._indexed_file

    def coordinated(self, normalized, path):
        entered.wait(timeout=10)
        return original(self, normalized, path)

    monkeypatch.setattr(ProjectIndex, "_indexed_file", coordinated)
    index = ProjectIndex(tmp_path)

    assert tuple(item.path for item in index.files) == ("a.java", "b.json")


def test_project_index_worker_override_fails_closed(monkeypatch):
    monkeypatch.setenv("MMM_PROJECT_INDEX_WORKERS", "0")
    with pytest.raises(ValueError, match="MMM_PROJECT_INDEX_WORKERS"):
        safety._project_index_scan_workers(2)
