from __future__ import annotations

import inspect
import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerator,
    _mutable_stage_state_sha256,
    _output_exhaustion_continuation_messages,
    _persist_generation_checkpoint,
    _prepare_generation_checkpoint,
    _read_generation_checkpoint_manifest,
    _remove_generation_checkpoint,
)


def test_output_continuation_carries_preserved_source_state() -> None:
    module = ProductionModule(
        module_id="example_block",
        kind="block",
        config={"implementation": "custom"},
    )
    messages = _output_exhaustion_continuation_messages(
        module=module,
        minecraft_version="1.21.1",
        loader="fabric",
        mappings="1.21.1+build.3",
        java_version=21,
        continuation_index=2,
        state_sha256="sha256:" + "a" * 64,
        touched_paths=("src/main/java/example/Block.java",),
        discarded_paths=("build.gradle",),
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    payload = json.loads(messages[1]["content"])
    continuation = payload["continuation"]
    assert continuation["reason"] == "previous_tool_enabled_page_exhausted_output"
    assert continuation["continuation_index"] == 2
    assert continuation["preserved_source_state_sha256"] == "sha256:" + "a" * 64
    assert continuation["preserved_paths_preview"] == [
        "src/main/java/example/Block.java"
    ]
    assert "resume with normal source/RAG tools" in messages[0]["content"]


def test_checkpoint_persists_staged_mutation_across_resume(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example/Example.java"
    source.parent.mkdir(parents=True)
    source.write_text("class Example { int before; }", encoding="utf-8")

    identity = "sha256:" + "b" * 64
    checkpoint_base = tmp_path / ".mmm-custom-checkpoints"
    checkpoint_root, staged_root, resumed, lease = _prepare_generation_checkpoint(
        root,
        identity_sha256=identity,
        configured_root=checkpoint_base,
    )
    try:
        assert resumed is False
        staged_source = staged_root / "src/main/java/example/Example.java"
        staged_source.write_text("class Example { int after; }", encoding="utf-8")
        expected_state = _mutable_stage_state_sha256(staged_root)
        _persist_generation_checkpoint(
            checkpoint_root,
            staged_root,
            identity_sha256=identity,
        )
        manifest = _read_generation_checkpoint_manifest(checkpoint_root)
        assert manifest["identity_sha256"] == identity
        assert manifest["stage_tree_sha256"]
    finally:
        lease.close()

    checkpoint_root_2, staged_root_2, resumed_2, lease_2 = _prepare_generation_checkpoint(
        root,
        identity_sha256=identity,
        configured_root=checkpoint_base,
    )
    try:
        assert checkpoint_root_2 == checkpoint_root
        assert resumed_2 is True
        assert (
            staged_root_2 / "src/main/java/example/Example.java"
        ).read_text(encoding="utf-8") == "class Example { int after; }"
        assert _mutable_stage_state_sha256(staged_root_2) == expected_state
    finally:
        lease_2.close()
        _remove_generation_checkpoint(checkpoint_root_2)


def test_output_boundary_loop_is_tool_enabled_checkpointed_and_fixed_point_bounded() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    exception_block = source[source.index("except BaseException as exc:") :]

    assert "enable_tools=True" in source
    assert "enable_tools=False" not in source
    assert exception_block.index("_persist_generation_checkpoint(") < exception_block.index(
        "boundary_kind = completion_boundary_kind(exc)"
    )
    assert "seen_output_states: set[str] = set()" in source
    assert "if state_sha256 in seen_output_states:" in source
    assert "Output continuation reached a no-source-progress fixed point." in source
    assert "seen_output_states.add(state_sha256)" in source
