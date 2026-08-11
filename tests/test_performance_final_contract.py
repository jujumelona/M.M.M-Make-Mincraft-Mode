from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.performance_final_contract import (
    StagedCommitConflict,
    _three_way_merge,
)
from minecraft_mod_ai.work_graph import _node


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
    assert deterministic.resource_class == "commit"
    assert asset.resource_class == "image_gpu"
    assert audio.resource_class == "cpu_io"
    assert finalize.resource_class == "commit"


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
