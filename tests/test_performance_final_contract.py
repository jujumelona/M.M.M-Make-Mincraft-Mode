from __future__ import annotations

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.performance_final_contract import (
    StagedCommitConflict,
    _acquire_wave_source_snapshot,
    _clone_source_snapshot,
    _clone_wave_workspace,
    _release_wave_source_snapshot,
    _three_way_merge,
)
from minecraft_mod_ai.work_graph import _module_stage, _node


def test_generation_nodes_use_resource_specific_lanes() -> None:
    custom = _node(
        "custom",
        "generate:custom",
        (),
        {"kind": "module-shard", "generation_stage": "custom"},
    )
    deterministic = _node(
        "content",
        "generate:content",
        (),
        {"kind": "module-shard", "generation_stage": "content"},
    )
    audio_binding = _node(
        "audio-binding",
        "generate:audio-binding",
        (),
        {"kind": "module-shard", "generation_stage": "audio-binding"},
    )
    asset = _node(
        "asset",
        "generate:assets",
        (),
        {"kind": "asset-shard"},
    )
    audio = _node(
        "audio",
        "generate:audio-synth",
        (),
        {"kind": "audio-synth"},
    )
    finalize = _node(
        "audio-finalize",
        "generate:audio-finalize",
        (),
        {"kind": "audio-finalize"},
    )

    assert custom.resource_class == "llm"
    assert deterministic.resource_class == "cpu_io"
    assert audio_binding.resource_class == "commit"
    assert asset.resource_class == "image_gpu"
    assert audio.resource_class == "cpu_io"
    assert finalize.resource_class == "commit"


def test_generic_integration_is_not_hidden_in_content_commit_lane() -> None:
    generic = ProductionModule(
        "custom_bridge",
        "integration",
        {"integration_type": "third_party_bridge"},
    )
    sidecar = ProductionModule(
        "local_sidecar",
        "integration",
        {"integration_type": "mmm_local_ai_sidecar"},
    )

    assert _module_stage(generic) == "custom"
    assert _module_stage(sidecar) == "content"


def test_local_gpu_text_role_participates_in_gpu_exclusion() -> None:
    config = ModelRegistry._resolve_role(
        "coder",
        {
            "model_id": "local/test-model",
            "adapter": "llama_cpp",
            "exclusive_gpu": False,
        },
    )
    cpu_config = ModelRegistry._resolve_role(
        "embedding",
        {
            "model_id": "local/test-embedding",
            "adapter": "embedding",
            "exclusive_gpu": False,
        },
    )

    assert config.exclusive_gpu is True
    assert cpu_config.exclusive_gpu is False


def test_custom_staging_clones_only_indexable_source_text(tmp_path: Path) -> None:
    project = tmp_path / "project"
    java = project / "src/main/java/example/Example.java"
    metadata = project / "src/main/resources/fabric.mod.json"
    image = project / "src/main/resources/assets/example/textures/item/icon.png"
    build_output = project / "build/generated/generated.txt"
    ai_receipt = project / ".minecraft_ai/large-receipt.json"
    for path in (java, metadata, image, build_output, ai_receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    java.write_text("class Example {}\n", encoding="utf-8")
    metadata.write_text('{"id":"example"}\n', encoding="utf-8")
    image.write_bytes(b"not-needed-binary")
    build_output.write_text("not source\n", encoding="utf-8")
    ai_receipt.write_text('{"ignored":true}\n', encoding="utf-8")

    stage = _clone_source_snapshot(project)

    assert (stage / java.relative_to(project)).is_file()
    assert (stage / metadata.relative_to(project)).is_file()
    assert not (stage / image.relative_to(project)).exists()
    assert not (stage / build_output.relative_to(project)).exists()
    assert not (stage / ai_receipt.relative_to(project)).exists()


def test_overlapping_custom_jobs_share_one_immutable_wave_snapshot(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src/main/java/example/Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Example {}\n", encoding="utf-8")

    first = _acquire_wave_source_snapshot(project)
    second = _acquire_wave_source_snapshot(project)
    assert first == second
    assert first.is_dir()

    workspace_a = _clone_wave_workspace(first, project)
    workspace_b = _clone_wave_workspace(second, project)
    staged_a = workspace_a / source.relative_to(project)
    staged_b = workspace_b / source.relative_to(project)
    staged_a.write_text("class Example { int a; }\n", encoding="utf-8")

    assert staged_b.read_text(encoding="utf-8") == "class Example {}\n"
    assert (first / source.relative_to(project)).read_text(encoding="utf-8") == (
        "class Example {}\n"
    )
    assert source.read_text(encoding="utf-8") == "class Example {}\n"

    import shutil

    shutil.rmtree(workspace_a, ignore_errors=True)
    shutil.rmtree(workspace_b, ignore_errors=True)
    _release_wave_source_snapshot(project, first)
    assert first.is_dir()
    _release_wave_source_snapshot(project, second)
    assert not first.exists()


def test_staged_java_merge_preserves_independent_insertions() -> None:
    base = "class Example {\n    void init() {\n    }\n}\n"
    staged = "class Example {\n    void init() {\n        registerCustom();\n    }\n}\n"
    live = "class Example {\n    void init() {\n        registerSystem();\n    }\n}\n"

    merged = _three_way_merge(
        "src/main/java/example/Example.java",
        base_text=base,
        staged_text=staged,
        live_text=live,
    )

    assert "registerCustom();" in merged
    assert "registerSystem();" in merged


def test_staged_json_merge_preserves_parallel_additions() -> None:
    base = json.dumps({"entrypoints": {"main": ["example.Base"]}}, indent=2) + "\n"
    staged = json.dumps(
        {"entrypoints": {"main": ["example.Base", "example.Custom"]}},
        indent=2,
    ) + "\n"
    live = json.dumps(
        {"entrypoints": {"main": ["example.Base", "example.System"]}},
        indent=2,
    ) + "\n"

    merged = json.loads(
        _three_way_merge(
            "src/main/resources/fabric.mod.json",
            base_text=base,
            staged_text=staged,
            live_text=live,
        )
    )

    assert merged["entrypoints"]["main"] == [
        "example.Base",
        "example.System",
        "example.Custom",
    ]


def test_staged_json_merge_rejects_same_key_semantic_conflict() -> None:
    base = json.dumps({"loader": "base"}) + "\n"
    staged = json.dumps({"loader": "custom"}) + "\n"
    live = json.dumps({"loader": "system"}) + "\n"

    with pytest.raises(StagedCommitConflict, match="Concurrent JSON merge conflict"):
        _three_way_merge(
            "src/main/resources/fabric.mod.json",
            base_text=base,
            staged_text=staged,
            live_text=live,
        )
