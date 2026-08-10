from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
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
        self.consumed_joint_contract = False

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "json"
        request = json.loads(messages[-1]["content"])
        assert request["phase"] == "generate_patch"
        self.requests.append(request)
        context = request["relevant_context"]
        anchor_text = "\n".join(
            item["text"] for item in context["global_anchors"]
        )
        assert "FIRST_PAGE_SOURCE_FACT" in anchor_text

        if context["page_index"] == 0 and context["page_count"] > 1:
            return json.dumps(
                {
                    "operations": [
                        {
                            "operation": "create",
                            "path": (
                                "src/main/resources/mmm_context_review/"
                                "inspection.json"
                            ),
                            "content": '{"inspection":"complete"}',
                        }
                    ],
                    "runtime_tests": [],
                    "complete": False,
                    "next_cursor": "",
                    "context_page_complete": True,
                }
            )
        if not context["complete"]:
            assert request["prior_patch_receipt"]["operation_count"] == 1
            return json.dumps(
                {
                    "operations": [],
                    "runtime_tests": [],
                    "complete": False,
                    "next_cursor": "",
                    "context_page_complete": True,
                }
            )

        expected_prior = 1 if context["page_count"] > 1 else 0
        assert request["prior_patch_receipt"]["operation_count"] == expected_prior
        self.consumed_joint_contract = True
        return json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "path": "src/main/java/example/GeneratedHook.java",
                        "content": (
                            "package example; final class GeneratedHook { "
                            'static final String FIRST = "FIRST_PAGE_SOURCE_FACT"; }\n'
                        ),
                    }
                ],
                "runtime_tests": ["Generated hook preserves the discovered source contract."],
                "complete": True,
                "next_cursor": "",
                "context_page_complete": True,
            }
        )


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
    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=budget),
    ).generate(root, module=module)

    generated = source / "GeneratedHook.java"
    assert result["status"] == "SOURCE_GENERATED"
    assert router.consumed_joint_contract is True
    assert router.requests
    assert all(
        _serialized_size(request["relevant_context"]) <= budget
        for request in router.requests
    )
    generated_text = generated.read_text(encoding="utf-8")
    assert result["operation_count"] >= 1
    assert "FIRST_PAGE_SOURCE_FACT" in generated_text
    assert result["source_observation_receipt"]["source_page_count"] >= 1
