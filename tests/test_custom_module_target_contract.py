from __future__ import annotations

from types import SimpleNamespace

import pytest

import minecraft_mod_ai.custom_module_generator as custom_generation
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import (
    CustomModuleGenerationError,
    CustomModuleGenerator,
)


class _ReachedProjectIndex(RuntimeError):
    pass


def _module() -> ProductionModule:
    return ProductionModule(module_id="target_probe", kind="custom_java", config={})


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
        raise _ReachedProjectIndex

    monkeypatch.setattr(custom_generation, "ProjectIndex", stop_after_target)
    generator = CustomModuleGenerator(object())

    with pytest.raises(_ReachedProjectIndex):
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
