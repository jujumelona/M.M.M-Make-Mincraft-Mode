from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import deterministic_minecraft_content_contract as contract


@dataclass
class _Module:
    module_id: str
    kind: str
    config: dict
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()

    def validate(self, *, policy=None) -> None:
        if self.config.get("invalid"):
            raise ValueError("invalid config")


class _Extended:
    _SUPPORTED = frozenset({"item", "block", "recipe", "loot"})
    ProductionModule = _Module


def test_compile_modules_preserves_strict_semantic_intent() -> None:
    modules = contract._compile_modules(
        _Extended,
        {
            "modules": [
                {
                    "id": "copper_hammer",
                    "kind": "item",
                    "config": {"display_name_en": "Copper Hammer"},
                    "depends_on": ["copper_ingot"],
                }
            ]
        },
    )

    assert len(modules) == 1
    assert modules[0].module_id == "copper_hammer"
    assert modules[0].kind == "item"
    assert modules[0].depends_on == ("copper_ingot",)
    assert modules[0].config == {"display_name_en": "Copper Hammer"}


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {"modules": [{"id": "CopperHammer", "kind": "item"}]},
            "Invalid strict Minecraft module id",
        ),
        (
            {
                "modules": [
                    {"id": "copper_hammer", "kind": "item"},
                    {"id": "copper_hammer", "kind": "block"},
                ]
            },
            "Duplicate Minecraft module id",
        ),
        (
            {"modules": [{"id": "copper_hammer", "kind": "entity"}]},
            "Unsupported deterministic Minecraft module kind",
        ),
        (
            {
                "modules": [
                    {
                        "id": "copper_hammer",
                        "kind": "item",
                        "path": "src/main/resources/evil.json",
                    }
                ]
            },
            "unknown fields",
        ),
        (
            {
                "modules": [
                    {
                        "id": "copper_hammer",
                        "kind": "item",
                        "depends_on": ["copper_hammer"],
                    }
                ]
            },
            "cannot depend on itself",
        ),
    ],
)
def test_compile_modules_fails_closed(payload: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        contract._compile_modules(_Extended, payload)


def test_schema_exposes_semantics_not_host_owned_paths() -> None:
    schema = contract._tool_schema(_Extended)
    function = schema["function"]
    parameters = function["parameters"]
    module_properties = parameters["properties"]["modules"]["items"]["properties"]

    assert function["name"] == "apply_minecraft_content_spec"
    assert set(module_properties) == {"id", "kind", "config", "depends_on"}
    assert "project_root" not in parameters["properties"]
    assert "mod_id" not in parameters["properties"]
    assert "package_name" not in parameters["properties"]
    assert "content" not in module_properties


def test_compact_record_never_returns_generated_file_bodies(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inside = project / "src/main/resources/assets/demo/lang/en_us.json"
    outside = tmp_path / "outside.txt"
    receipt = contract._compact_record(
        {
            "status": "GENERATED",
            "modules": ["copper_hammer"],
            "catalog_module_count": 3,
            "files": [str(inside), str(outside)],
            "source_receipt": {"content": "must-not-leak"},
            "binding_receipt": {"source": "must-not-leak"},
            "required_gates": ["JDT", "Gradle"],
        },
        project_root=project,
        requested_module_ids=["copper_hammer"],
    )

    assert receipt["status"] == "GENERATED"
    assert receipt["module_ids"] == ["copper_hammer"]
    assert receipt["touched_paths"] == [
        "src/main/resources/assets/demo/lang/en_us.json"
    ]
    assert "source_receipt" not in receipt
    assert "binding_receipt" not in receipt
    assert "must-not-leak" not in repr(receipt)


def test_execute_uses_host_discovered_project_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "demo"
    (project / "src").mkdir(parents=True)
    (project / "build.gradle").write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    class _Runtime:
        class AgentToolRuntimeError(RuntimeError):
            pass

        @staticmethod
        def _discover_model_project_root(workspace_root):
            assert Path(workspace_root) == tmp_path
            return project, "demo"

    class _GeneratingExtended(_Extended):
        @staticmethod
        def generate_extended_content(**kwargs):
            captured.update(kwargs)
            return {
                "status": "GENERATED",
                "modules": [module.module_id for module in kwargs["modules"]],
                "files": [str(project / "src/main/resources/generated.json")],
                "required_gates": ["JDT", "Gradle"],
            }

    monkeypatch.setattr(
        "minecraft_mod_ai.project_edit.inspect_fabric_project",
        lambda root: SimpleNamespace(
            root=Path(root),
            mod_id="demo_mod",
            package_name="com.example.demo",
        ),
    )

    receipt = contract._execute(
        _Runtime,
        _GeneratingExtended,
        tmp_path,
        {"modules": [{"id": "copper_hammer", "kind": "item"}]},
    )

    assert captured["project_root"] == project
    assert captured["mod_id"] == "demo_mod"
    assert captured["package_name"] == "com.example.demo"
    assert receipt["module_ids"] == ["copper_hammer"]


def test_role_policy_exposes_both_small_model_host_tools() -> None:
    class _Capability:
        @staticmethod
        def _policy_model_role(stage, model_role):
            return model_role

        @staticmethod
        def skills_for_model_role(model_role):
            assert model_role == "coder"
            return frozenset({"generate-datagen", "patch-existing-project"})

    assert contract._role_dynamic_tools(_Capability, "generation", "coder") == {
        "apply_minecraft_content_spec": "generate-datagen",
        "apply_source_edit": "patch-existing-project",
    }
    assert contract._role_dynamic_tools(_Capability, "quality", "coder") == {}
