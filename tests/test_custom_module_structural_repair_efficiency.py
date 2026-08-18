from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.scale_policy import ScalePolicy


class _StructuralRepairRouter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages: list[list[dict[str, str]]] = []

    def bind_agent_workspace(self, *_args, **_kwargs):
        return self

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        self.calls.append(dict(kwargs))
        self.messages.append([dict(message) for message in messages])
        if len(self.calls) == 1:
            return json.dumps({"runtime_tests": [], "complete": True})
        return json.dumps(
            {
                "operations": [
                    {
                        "operation": "create",
                        "path": "src/main/java/example/Generated.java",
                        "content": "package example; final class Generated {}\n",
                    }
                ],
                "runtime_tests": [],
                "complete": True,
                "next_cursor": "",
                "context_page_complete": True,
            }
        )


def test_structural_response_repair_never_enables_rag_or_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def forbidden_index(*_args, **_kwargs):
        raise AssertionError("structural repair must not build project RAG")

    from minecraft_mod_ai.production_tools import ProductionToolService

    monkeypatch.setattr(ProductionToolService, "index_project_rag", forbidden_index)
    router = _StructuralRepairRouter()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(
        router,
        policy=ScalePolicy(model_context_bytes=4096),
    ).generate(
        root,
        module=ProductionModule("shape_repair", "custom_java", {"feature": "shape"}),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )

    assert result["status"] == "SOURCE_GENERATED"
    assert len(router.calls) == 2
    assert all(call.get("enable_tools") is False for call in router.calls)
    assert (root / "src/main/java/example/Generated.java").is_file()

    initial = router.messages[0]
    repaired = router.messages[1]
    assert len(initial) == 2
    assert repaired[:2] == initial
    assert len(repaired) == 3
    assert repaired[2]["role"] == "user"
    assert "Repair only the JSON/patch/cursor transition" in repaired[2]["content"]
    assert "invalid assistant payload is intentionally omitted" in repaired[2]["content"]
    assert "Do not retrieve new RAG/MCP evidence" in repaired[2]["content"]
