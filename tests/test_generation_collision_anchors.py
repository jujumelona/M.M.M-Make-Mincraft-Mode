from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import generation_concurrency_safety as safety


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


def test_builtin_generators_keep_shared_collision_domains():
    content = SimpleNamespace(kind="item")
    integration = SimpleNamespace(kind="integration")
    entity = SimpleNamespace(kind="entity")
    system = SimpleNamespace(kind="quest")

    assert safety._builtin_shared_anchors(content, "content")
    assert safety._builtin_shared_anchors(integration, "content") == ()
    assert safety._builtin_shared_anchors(entity, "entity")
    assert safety._builtin_shared_anchors(system, "system")
