from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.scale_policy import ScalePolicy


def _serialized_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _latest_json_request(messages) -> dict:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            request = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(request, dict) and request.get("phase"):
            return request
    raise AssertionError("No structured coder request was found in the message history.")


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


class _ContextPagingRouter:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.file_contexts: list[dict] = []
        self.consumed_joint_contract = False

    def bind_agent_workspace(self, *_args, **_kwargs):
        return self

    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("custom-module generation must use native structured return channels")

    def generate_tool_decision(
        self,
        role,
        messages,
        *,
        tool_name,
        parameters,
        description="",
    ):
        del description
        assert role == "coder"
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        request = _latest_json_request(messages)
        self.requests.append(request)

        if tool_name == "return_custom_module_file_plan":
            assert request["phase"] == "plan_files"
            assert _serialized_size(request["planning_context"]) <= 4096
            return {
                "files": [
                    {
                        "path": "src/main/java/example/GeneratedHook.java",
                        "purpose": (
                            "Implement crossFileHook while preserving FIRST_PAGE_SOURCE_FACT "
                            "and HIGH_INDEX_SOURCE_SENTINEL exact-source contracts."
                        ),
                    }
                ],
                "runtime_tests": [],
            }

        if tool_name == "return_custom_module_file_content":
            assert request["phase"] == "write_file"
            context = request["exact_source_context"]
            self.file_contexts.append(context)
            assert _serialized_size(context) <= 4096
            observed_text = "\n".join(item["text"] for item in context["records"])
            assert "FIRST_PAGE_SOURCE_FACT" in observed_text
            assert "HIGH_INDEX_SOURCE_SENTINEL" in observed_text
            self.consumed_joint_contract = True
            return {
                "content": (
                    "package example; final class GeneratedHook { "
                    'static final String FIRST = "FIRST_PAGE_SOURCE_FACT"; '
                    'static final String LAST = "HIGH_INDEX_SOURCE_SENTINEL"; }\n'
                ),
                "runtime_tests": [
                    "Generated hook preserves source contracts discovered across host pages."
                ],
            }

        raise AssertionError(f"unexpected structured return channel: {tool_name}")


def test_custom_generator_consumes_relevant_source_beyond_first_context_page(
    tmp_path: Path,
) -> None:
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
        (
            "package example; // required contract FIRST_PAGE_SOURCE_FACT\n"
            "final class A0000 { static final String HOOK = \"crossFileHook\"; }\n"
        ),
    )
    for index in range(1, 60):
        write_equal_size(
            source / f"A{index:04d}.java",
            (
                f"package example; final class A{index:04d} {{ "
                'static final String HOOK = "crossFileHook"; }\n'
            ),
        )
    high_index = source / "Z9999.java"
    write_equal_size(
        high_index,
        (
            "package example; // required contract HIGH_INDEX_SOURCE_SENTINEL\n"
            "final class Z9999 { "
            'static final String HOOK = "crossFileHook"; '
            "}\n"
        ),
    )

    budget = 4096
    router = _ContextPagingRouter()
    module = ProductionModule(
        "cross_file_feature",
        "custom_java",
        {"feature": "crossFileHook"},
    )
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=budget),
    ).generate(
        root,
        module=module,
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    generated = source / "GeneratedHook.java"
    assert result["status"] == "SOURCE_GENERATED"
    assert router.consumed_joint_contract is True
    assert router.file_contexts
    assert all(_serialized_size(context) <= budget for context in router.file_contexts)
    generated_text = generated.read_text(encoding="utf-8")
    assert result["operation_count"] == 1
    assert "FIRST_PAGE_SOURCE_FACT" in generated_text
    assert "HIGH_INDEX_SOURCE_SENTINEL" in generated_text
    assert result["source_observation_receipt"]["source_page_count"] > 1
