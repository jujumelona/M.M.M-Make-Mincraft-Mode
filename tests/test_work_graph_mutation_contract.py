from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_contract_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "minecraft_mod_ai" / "work_graph_mutation_contract.py"
    spec = importlib.util.spec_from_file_location("_work_graph_mutation_contract_test_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_stage_wrapper_preserves_keyword_only_contract_and_forwards_kinds() -> None:
    contract = _load_contract_module()
    seen: dict[str, Any] = {}

    def original_stage(
        module: Any,
        *,
        deterministic_module_kinds: frozenset[str] | None = None,
    ) -> str:
        seen["module"] = module
        seen["deterministic_module_kinds"] = deterministic_module_kinds
        return "content"

    def original_node(
        node_id: str,
        stage: str,
        dependencies: tuple[str, ...],
        payload: dict[str, Any],
    ) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
        return node_id, stage, dependencies, payload

    work_graph_module = SimpleNamespace(
        _module_stage=original_stage,
        _node=original_node,
        is_research_shard=lambda module: False,
    )
    contract.install(work_graph_module)

    physical_signature = inspect.signature(
        work_graph_module._module_stage,
        follow_wrapped=False,
    )
    parameter = physical_signature.parameters["deterministic_module_kinds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None

    deterministic_module_kinds = frozenset({"recipe", "loot"})
    production_module = SimpleNamespace(kind="content", config={})
    assert work_graph_module._module_stage(
        production_module,
        deterministic_module_kinds=deterministic_module_kinds,
    ) == "content"
    assert seen["module"] is production_module
    assert seen["deterministic_module_kinds"] is deterministic_module_kinds


def test_node_wrapper_keeps_existing_contract_shape() -> None:
    contract = _load_contract_module()

    def original_stage(
        module: Any,
        *,
        deterministic_module_kinds: frozenset[str] | None = None,
    ) -> str:
        return "content"

    def original_node(
        node_id: str,
        stage: str,
        dependencies: tuple[str, ...],
        payload: dict[str, Any],
    ) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
        return node_id, stage, dependencies, payload

    work_graph_module = SimpleNamespace(
        _module_stage=original_stage,
        _node=original_node,
        is_research_shard=lambda module: False,
    )
    contract.install(work_graph_module)

    physical_signature = inspect.signature(work_graph_module._node, follow_wrapped=False)
    assert tuple(physical_signature.parameters) == (
        "node_id",
        "stage",
        "dependencies",
        "payload",
    )

    result = work_graph_module._node(
        "node",
        "generate:custom",
        (),
        {"kind": "module-shard", "generation_stage": "custom"},
    )
    assert result[3]["resource_class"] == "llm"
