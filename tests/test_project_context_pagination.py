from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.scale_policy import ScalePolicy


def _serialized_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_project_context_pages_reconstruct_large_utf8_source_within_budget(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example"
    source.mkdir(parents=True)
    original = "package example;\n" + "// 항해 시스템 crossFileHook\n" * 240 + "final class LargeNavigationSystem {}\n"
    target = source / "LargeNavigationSystem.java"
    target.write_text(original, encoding="utf-8")
    expected_on_disk = target.read_bytes().decode("utf-8")

    index = ProjectIndex(root)
    cursor = ""
    fragments: list[str] = []
    seen_cursors: set[str] = set()
    pages = 0
    while True:
        page = index.select_page(query="crossFileHook navigation", byte_budget=1400, cursor=cursor)
        pages += 1
        assert _serialized_size(page) <= 1400
        fragments.extend(item["content"] for item in page["files"] if item["path"] == target.relative_to(root).as_posix())
        cursor = page["next_cursor"]
        if not cursor:
            assert page["complete"] is True
            break
        assert cursor not in seen_cursors
        seen_cursors.add(cursor)

    assert pages > 1
    assert "".join(fragments) == expected_on_disk


class _ContextPagingRouter:
    def __init__(self) -> None:
        self.workspace: Path | None = None
        self.requests: list[dict] = []
        self.consumed_joint_contract = False

    def bind_agent_workspace(self, workspace_root, **_kwargs):
        self.workspace = Path(workspace_root)
        return self

    def generate_tool_decision(self, *_args, **_kwargs):
        raise AssertionError("custom-module generation must not use a file-plan return channel")

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "text"
        assert kwargs["tool_stage"] == "generation"
        assert kwargs["enable_tools"] is True
        assert self.workspace is not None
        request = json.loads(messages[-1]["content"])
        assert request["phase"] == "implement_module"
        context = request["initial_exact_source_context"]
        self.requests.append(request)
        assert _serialized_size(context) <= 4096
        observed = [*context["global_anchors"], *context["page_observations"]]
        observed_text = "\n".join(item["text"] for item in observed)
        assert "FIRST_PAGE_SOURCE_FACT" in observed_text
        assert "HIGH_INDEX_SOURCE_SENTINEL" in observed_text
        self.consumed_joint_contract = True

        project = self.workspace / request["workspace_project_root"]
        target = project / "src/main/java/example/GeneratedHook.java"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "package example; final class GeneratedHook { "
            'static final String FIRST = "FIRST_PAGE_SOURCE_FACT"; '
            'static final String LAST = "HIGH_INDEX_SOURCE_SENTINEL"; }\n',
            encoding="utf-8",
        )
        return "Implemented cross-file hook from grounded source evidence."


def test_custom_generator_exposes_grounded_context_and_agent_can_fetch_more_with_tools(tmp_path: Path) -> None:
    root = tmp_path / "project"
    source = root / "src/main/java/example"
    source.mkdir(parents=True)
    source_size = 320

    def write_equal_size(path: Path, body: str) -> None:
        encoded = body.encode("utf-8")
        assert len(encoded) < source_size
        path.write_bytes(encoded + b" " * (source_size - len(encoded)))

    write_equal_size(
        source / "A0000.java",
        "package example; // required contract FIRST_PAGE_SOURCE_FACT\n"
        'final class A0000 { static final String HOOK = "crossFileHook"; }\n',
    )
    for index in range(1, 60):
        write_equal_size(
            source / f"A{index:04d}.java",
            f"package example; final class A{index:04d} {{ "
            'static final String HOOK = "crossFileHook"; }\n',
        )
    write_equal_size(
        source / "Z9999.java",
        "package example; // required contract HIGH_INDEX_SOURCE_SENTINEL\n"
        'final class Z9999 { static final String HOOK = "crossFileHook"; }\n',
    )

    budget = 4096
    router = _ContextPagingRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(router, policy=ScalePolicy(model_context_bytes=budget)).generate(
        root,
        module=ProductionModule("cross_file_feature", "custom_java", {"feature": "crossFileHook"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    generated = source / "GeneratedHook.java"
    assert result["status"] == "SOURCE_GENERATED"
    assert router.consumed_joint_contract is True
    assert len(router.requests) == 1
    generated_text = generated.read_text(encoding="utf-8")
    assert result["operation_count"] == 1
    assert "FIRST_PAGE_SOURCE_FACT" in generated_text
    assert "HIGH_INDEX_SOURCE_SENTINEL" in generated_text
    assert result["source_observation_receipt"]["source_page_count"] > 1
