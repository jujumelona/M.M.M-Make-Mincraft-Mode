from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.custom_module_generator as generator_module
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.implementation_ir import OutputBudgetExhausted
from minecraft_mod_ai.llama_finish_reason_contract import (
    CONTEXT_PRESSURE,
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.model_adapters.base import ModelBackendError
from minecraft_mod_ai.platform_catalog import adapter_for_target


def _task_module(name: str = "Generated") -> ProductionModule:
    anchor = {
        "kind": "symbol",
        "locator": f"src/main/java/example/{name}.java#{name}",
        "status": "host_reserved",
        "source_set": "main",
    }
    target = adapter_for_target("1.20.1", "fabric")
    return ProductionModule(
        module_id=name.casefold(),
        kind="custom_java",
        config={
            "evidence_task": {
                "task_id": name.casefold(),
                "semantic_outcome": f"Implement {name}.",
                "target_cell": {
                    "minecraft_version": target.minecraft_version,
                    "loader": target.loader,
                    "mappings": target.yarn_mappings,
                    "java_version": target.java_version,
                },
                "owned_anchors": [anchor],
                "implementation_obligations": [f"Implement {name} exactly."],
                "production_bindings": [
                    {
                        "task_ref": name.casefold(),
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
                "required_gates": ["source_static_validation", "target_compile"],
            }
        },
    )


def _source(name: str = "Generated", *, value: int = 1) -> str:
    return (
        "package example;\n\n"
        f"public final class {name} {{\n"
        f"    private {name}() {{}}\n"
        f"    public static int value() {{ return {value}; }}\n"
        "}\n"
    )


class _Router:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], dict]] = []
        self.workspace: Path | None = None

    def bind_agent_workspace(self, workspace_root, **_kwargs):
        self.workspace = Path(workspace_root)
        return self

    def generate_text(self, role, messages, **kwargs):
        assert role == "coder"
        assert kwargs["response_format"] == "text"
        assert "response_schema" not in kwargs
        assert kwargs["enable_tools"] is False
        assert kwargs["tool_stage"] == "generation"
        self.calls.append(([dict(message) for message in messages], dict(kwargs)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _Runner:
    reports = []

    def __init__(self, _cache):
        pass

    def compile_java(self, _root):
        if not self.reports:
            return SimpleNamespace(status="PASS", commands=(), error=None)
        return self.reports.pop(0)


@pytest.fixture(autouse=True)
def _runner(monkeypatch):
    _Runner.reports = []
    monkeypatch.setattr(generator_module, "GradleRunner", _Runner)


def _generate(tmp_path: Path, router: _Router, *, name: str = "Generated"):
    root = tmp_path / "project"
    root.mkdir()
    target = adapter_for_target("1.20.1", "fabric")
    result = CustomModuleGenerator(router).generate(
        root,
        module=_task_module(name),
        minecraft_version=target.minecraft_version,
        loader=target.loader,
        mappings=target.yarn_mappings,
    )
    return root, result


def test_direct_coder_returns_whole_file_and_host_writes_it(tmp_path: Path) -> None:
    router = _Router([
        _source()
    ])
    root, result = _generate(tmp_path, router)

    target = root / "src/main/java/example/Generated.java"
    assert result["status"] == "SOURCE_GENERATED"
    assert result["touched_paths"] == ["src/main/java/example/Generated.java"]
    assert result["operation_count"] == 1
    assert result["generation_verification"]["attempt"] == 1
    assert "generation_checkpoint" not in result
    assert target.read_text(encoding="utf-8") == _source()
    assert len(router.calls) == 1


def test_compiler_failure_repair_uses_exact_failure_then_passes(tmp_path: Path) -> None:
    _Runner.reports = [
        SimpleNamespace(
            status="FAIL",
            commands=(),
            error="Generated.java:4: error: synthetic compiler failure",
        ),
        SimpleNamespace(status="PASS", commands=(), error=None),
    ]
    router = _Router([
        _source(value=1),
        _source(value=2),
    ])

    root, result = _generate(tmp_path, router)

    assert result["generation_verification"]["attempt"] == 2
    assert "synthetic compiler failure" in router.calls[1][0][-1]["content"]
    assert "not a patch" in router.calls[1][0][-1]["content"]
    assert "return 2" in (root / "src/main/java/example/Generated.java").read_text()


def test_non_improving_invariant_repair_stops_without_blind_retries(
    tmp_path: Path,
) -> None:
    broken_one = "package example; public final class Wrong {}"
    broken_two = "package example; public final class StillWrong {}"
    router = _Router([
        broken_one,
        broken_two,
    ])

    with pytest.raises(CustomModuleGenerationError, match="ceased to improve"):
        _generate(tmp_path, router)

    assert len(router.calls) == 2
    assert not (tmp_path / "project/src/main/java/example/Generated.java").exists()


def test_output_exhaustion_returns_to_graph_decomposition_and_rolls_back(
    tmp_path: Path,
) -> None:
    boundary = LlamaCompletionBoundaryError(
        "exhausted",
        kind=OUTPUT_EXHAUSTED,
        completion_tokens=8192,
        max_tokens=8192,
    )
    router = _Router([
        ModelBackendError(role="coder", model_id="test", cause=boundary)
    ])

    with pytest.raises(OutputBudgetExhausted, match="OUTPUT_BUDGET_EXHAUSTED"):
        _generate(tmp_path, router)

    assert len(router.calls) == 1
    assert not (tmp_path / "project/src/main/java/example/Generated.java").exists()


def test_context_pressure_remains_typed_backend_failure_and_rolls_back(
    tmp_path: Path,
) -> None:
    boundary = LlamaCompletionBoundaryError(
        "context pressure",
        kind=CONTEXT_PRESSURE,
        prompt_tokens=32000,
        max_tokens=8192,
    )
    failure = ModelBackendError(role="coder", model_id="test", cause=boundary)
    router = _Router([failure])

    with pytest.raises(ModelBackendError) as caught:
        _generate(tmp_path, router)

    assert caught.value is failure
    assert len(router.calls) == 1
    assert not (tmp_path / "project/src/main/java/example/Generated.java").exists()
