from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import minecraft_mod_ai.custom_module_generator as generator_module
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.project_index import ProjectIndex


def _serialized_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_project_context_pages_reconstruct_large_utf8_source_within_budget(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example"
    source.mkdir(parents=True)
    original = (
        "package example;\n"
        + "// 항해 시스템 crossFileHook\n" * 240
        + "final class LargeNavigationSystem {}\n"
    )
    target = source / "LargeNavigationSystem.java"
    target.write_text(original, encoding="utf-8")
    expected_on_disk = target.read_bytes().decode("utf-8")

    index = ProjectIndex(root)
    cursor = ""
    fragments: list[str] = []
    seen_cursors: set[str] = set()
    pages = 0
    while True:
        page = index.select_page(
            query="crossFileHook navigation",
            byte_budget=1400,
            cursor=cursor,
        )
        pages += 1
        assert _serialized_size(page) <= 1400
        fragments.extend(
            item["content"]
            for item in page["files"]
            if item["path"] == target.relative_to(root).as_posix()
        )
        cursor = page["next_cursor"]
        if not cursor:
            assert page["complete"] is True
            break
        assert cursor not in seen_cursors
        seen_cursors.add(cursor)

    assert pages > 1
    assert "".join(fragments) == expected_on_disk


class _ContextRouter:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.messages: list[list[dict[str, str]]] = []

    def bind_agent_workspace(self, workspace_root, **_kwargs):
        self.workspace = Path(workspace_root)
        return self

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "json"
        assert kwargs["tool_stage"] == "generation"
        assert kwargs["enable_tools"] is False
        assert self.workspace is not None
        copied = [dict(message) for message in messages]
        self.messages.append(copied)
        prompt = "\n".join(str(message.get("content", "")) for message in copied)
        assert "FIRST_PAGE_SOURCE_FACT" in prompt
        assert "HIGH_INDEX_SOURCE_SENTINEL" in prompt
        assert "A0001" not in prompt

        source = (
            "package example;\n\n"
            "public final class GeneratedHook {\n"
            "    private GeneratedHook() {}\n"
            '    static final String FIRST = "FIRST_PAGE_SOURCE_FACT";\n'
            '    static final String LAST = "HIGH_INDEX_SOURCE_SENTINEL";\n'
            "}\n"
        )
        return json.dumps(
            {
                "content": source,
                "summary": "Implemented from host-selected relevant source.",
            }
        )


def test_custom_generator_sends_only_relevant_existing_source_to_whole_file_coder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example"
    source.mkdir(parents=True)

    (source / "A0000.java").write_text(
        "package example; // FIRST_PAGE_SOURCE_FACT\n"
        "final class A0000 {}\n",
        encoding="utf-8",
    )
    for index in range(1, 60):
        (source / f"A{index:04d}.java").write_text(
            f"package example; final class A{index:04d} {{}}\n",
            encoding="utf-8",
        )
    (source / "Z9999.java").write_text(
        "package example; // HIGH_INDEX_SOURCE_SENTINEL\n"
        "final class Z9999 {}\n",
        encoding="utf-8",
    )

    class Runner:
        def __init__(self, _cache):
            pass

        def compile_java(self, _root):
            return SimpleNamespace(status="PASS", commands=(), error=None)

    monkeypatch.setattr(generator_module, "GradleRunner", Runner)

    router = _ContextRouter()
    target = adapter_for_target("1.20.1", "fabric")
    anchor = {
        "kind": "symbol",
        "locator": "src/main/java/example/GeneratedHook.java#GeneratedHook",
        "status": "host_reserved",
        "source_set": "main",
    }
    result = CustomModuleGenerator(router).generate(
        root,
        module=ProductionModule(
            "cross_file_feature",
            "custom_java",
            {
                "feature": "crossFileHook",
                "evidence_task": {
                    "task_id": "cross_file_feature",
                    "semantic_outcome": (
                        "Implement GeneratedHook using the A0000 and Z9999 contracts."
                    ),
                    "target_cell": {
                        "minecraft_version": target.minecraft_version,
                        "loader": target.loader,
                        "mappings": target.yarn_mappings,
                        "java_version": target.java_version,
                    },
                    "owned_anchors": [anchor],
                    "implementation_obligations": [
                        "Preserve facts from A0000 and Z9999 in GeneratedHook."
                    ],
                    "production_bindings": [
                        {
                            "task_ref": "cross_file_feature",
                            "reuse_action": "fresh",
                            "owned_anchors": [anchor],
                        }
                    ],
                    "required_gates": [
                        "source_static_validation",
                        "target_compile",
                    ],
                },
            },
        ),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    generated = source / "GeneratedHook.java"
    assert result["status"] == "SOURCE_GENERATED"
    assert len(router.messages) == 1
    generated_text = generated.read_text(encoding="utf-8")
    assert result["operation_count"] == 1
    assert "FIRST_PAGE_SOURCE_FACT" in generated_text
    assert "HIGH_INDEX_SOURCE_SENTINEL" in generated_text
    receipt = result["source_observation_receipt"]
    assert receipt["path"] == "src/main/java/example/GeneratedHook.java"
    assert receipt["sha256"].startswith("sha256:")
