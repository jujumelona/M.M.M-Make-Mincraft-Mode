from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.custom_module_generator as custom_generation
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
)


class _ReachedScaffold(RuntimeError):
    pass


def _module(*, minecraft_version: str = "26.2", mappings: str = "", java_version: int = 25) -> ProductionModule:
    anchor = {
        "kind": "symbol",
        "locator": "src/main/java/example/TargetProbe.java#TargetProbe",
        "status": "host_reserved",
        "source_set": "main",
    }
    return ProductionModule(
        module_id="target_probe",
        kind="custom_java",
        config={
            "evidence_task": {
                "task_id": "target_probe",
                "semantic_outcome": "Exercise the generation target boundary.",
                "target_cell": {
                    "minecraft_version": minecraft_version,
                    "loader": "fabric",
                    "mappings": mappings,
                    "java_version": java_version,
                },
                "owned_anchors": [anchor],
                "implementation_obligations": [
                    "Materialize the owned TargetProbe type at the exact host-reserved target."
                ],
                "production_bindings": [
                    {
                        "task_ref": "target_probe",
                        "reuse_action": "fresh",
                        "owned_anchors": [anchor],
                    }
                ],
                "required_gates": ["source_static_validation"],
            }
        },
    )


def test_native_26_2_blank_mappings_crosses_generation_target_gate(monkeypatch, tmp_path) -> None:
    adapter = SimpleNamespace(
        minecraft_version="26.2",
        loader="fabric",
        yarn_mappings="",
        java_version="25",
        mappings_applicable=False,
    )
    monkeypatch.setattr(custom_generation, "adapter_for_target", lambda version, loader: adapter)

    def stop_after_target(*_args, **_kwargs):
        raise _ReachedScaffold

    monkeypatch.setattr(custom_generation, "_materialize_host_scaffold", stop_after_target)
    generator = CustomModuleGenerator(object())

    with pytest.raises(_ReachedScaffold):
        generator.generate(
            tmp_path,
            module=_module(),
            minecraft_version="26.2",
            loader="fabric",
            mappings="",
        )


def test_mapped_target_still_requires_mapping_at_generation_boundary(monkeypatch, tmp_path) -> None:
    adapter = SimpleNamespace(
        minecraft_version="1.21.4",
        loader="fabric",
        yarn_mappings="1.21.4+build.8",
        java_version="21",
        mappings_applicable=True,
    )
    monkeypatch.setattr(custom_generation, "adapter_for_target", lambda version, loader: adapter)
    generator = CustomModuleGenerator(object())

    with pytest.raises(CustomModuleGenerationError, match="TARGET_MAPPINGS_REQUIRED"):
        generator.generate(
            tmp_path,
            module=_module(),
            minecraft_version="1.21.4",
            loader="fabric",
            mappings="",
        )
