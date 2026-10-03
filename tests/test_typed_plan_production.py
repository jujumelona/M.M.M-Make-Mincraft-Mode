from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.work_graph import _is_host_exact_authored_module, _module_shards, _node



class ForbiddenRouter:
    def __getattr__(self, name):
        raise AssertionError(f"Typed PlanIR production touched router attribute {name!r}")


def _project(root: Path) -> Path:
    spec = ModSpec(
        mod_id="typedtest",
        mod_name="Typed Test",
        package_name="ai.minecraft.typedtest",
        version="1.0.0",
        summary="typed plan test",
        contents=(
            ContentSpec(
                content_id="marker_item",
                kind=ContentKind.ITEM,
                display_name_en="Marker",
                display_name_ko="마커",
            ),
        ),
    )
    FabricProjectGenerator().generate(spec, root)
    return root


def _plan(text: str = "typed behavior") -> dict:
    return {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "compute",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {"op": "literal", "type": "int", "value": 42},
                    }
                ],
                "covers": ["design:compute"],
            }
        ],
        "initialize": [],
    }


def test_typed_plan_backend_generates_compileable_java_without_router(tmp_path: Path) -> None:
    root = _project(tmp_path)
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="custom_java",
        config={
            "implementation": "custom",
            "typed_plan_ir": _plan(),
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
        },
        required_gates=("target_compile",),
    )

    receipt = CustomModuleGenerator(ForbiddenRouter()).generate(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    source_path = (
        root / "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "// MMM:TYPED_PLAN_OWNER" in source
    assert "return 42;" in source

    javac = shutil.which("javac")
    if not javac:
        pytest.skip("JDK required")
    classes = root / ".typed-test-classes"
    classes.mkdir()
    compiled = subprocess.run(
        [javac, "-encoding", "UTF-8", "-d", str(classes), str(source_path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_typed_state_backend_generates_state_owner_without_router(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    text = "stateful behavior"
    typed = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "readCoins",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {
                            "op": "state_get",
                            "key": {
                                "op": "literal",
                                "type": "string",
                                "value": "coins",
                            },
                            "type": "int",
                            "context": {"op": "map", "entries": []},
                        },
                    }
                ],
                "covers": ["state:coins"],
            }
        ],
        "initialize": [],
    }
    state = {
        "specification": {
            "variables": [
                {
                    "name": "coins",
                    "owner": "player",
                    "type": "integer",
                    "unit": "credits",
                    "default": "0",
                    "domain": "integer >= 0",
                }
            ],
            "transitions": [],
            "invariants": [],
            "initialization": [],
            "updates": [],
            "cleanup": [],
            "concurrency": [],
            "inapplicable_concerns": [],
        },
        "constraint_evidence_refs": [],
    }
    module = ProductionModule(
        module_id="authored_typed_state",
        kind="custom_java",
        config={
            "implementation": "custom",
            "typed_plan_ir": typed,
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
            "typed_plan_state_section": state,
        },
        required_gates=("target_compile",),
    )

    receipt = CustomModuleGenerator(ForbiddenRouter()).generate(
        root,
        module=module,
    )

    assert receipt["generation_verification"]["model_calls"] == 0
    package_root = root / "src/main/java/ai/minecraft/typedtest"
    program = package_root / "AuthoredProgram.java"
    state_owner = package_root / "AuthoredStateModel.java"
    assert program.is_file()
    assert state_owner.is_file()
    assert "// MMM:TYPED_PLAN_STATE_OWNER" in state_owner.read_text(encoding="utf-8")

    javac = shutil.which("javac")
    if not javac:
        pytest.skip("JDK required")
    classes = root / ".typed-state-test-classes"
    classes.mkdir()
    compiled = subprocess.run(
        [
            javac,
            "-encoding",
            "UTF-8",
            "-d",
            str(classes),
            str(state_owner),
            str(program),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_typed_state_operations_fail_closed_without_canonical_state_authority() -> None:
    text = "stateful behavior"
    typed = {
        "schema_version": "mmm/typed-plan-ir-v1",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "functions": [
            {
                "id": "readCoins",
                "parameters": [],
                "return_type": "int",
                "body": [
                    {
                        "op": "return",
                        "value": {
                            "op": "state_get",
                            "key": {
                                "op": "literal",
                                "type": "string",
                                "value": "coins",
                            },
                            "type": "int",
                            "context": {
                                "op": "map",
                                "entries": [],
                            },
                        },
                    }
                ],
                "covers": ["state:coins"],
            }
        ],
        "initialize": [],
    }
    plan = AuthoredPlan(
        requested_prompt="stateful mod",
        text=text,
        typed_plan_ir=typed,
    )

    with pytest.raises(ValueError, match="TYPED_PLAN_STATE_AUTHORITY_REQUIRED"):
        _compile_new_authored_modules(
            plan,
            mod_id="typedtest",
            package_name="ai.minecraft.typedtest",
            target={},
            production_state_section={},
        )

def test_typed_plan_backend_refuses_unowned_existing_source(tmp_path: Path) -> None:
    root = _project(tmp_path)
    source_path = (
        root / "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
    )
    source_path.write_text(
        "package ai.minecraft.typedtest; public final class AuthoredProgram {}\n",
        encoding="utf-8",
    )
    module = ProductionModule(
        module_id="authored_typed_plan",
        kind="custom_java",
        config={
            "implementation": "custom",
            "typed_plan_ir": _plan(),
            "typed_plan_package": "ai.minecraft.typedtest",
            "typed_plan_path": (
                "src/main/java/ai/minecraft/typedtest/AuthoredProgram.java"
            ),
            "typed_plan_capabilities": {},
        },
        required_gates=("target_compile",),
    )

    with pytest.raises(ValueError, match="TYPED_PLAN_OWNERSHIP_CONFLICT"):
        CustomModuleGenerator(ForbiddenRouter()).generate(
            root,
            module=module,
        )


def test_typed_plan_host_work_is_isolated_from_llm_shards() -> None:
    typed = ProductionModule(
        module_id="typed",
        kind="custom_java",
        config={
            "implementation": "custom",
            "typed_plan_ir": _plan(),
        },
    )
    legacy = ProductionModule(
        module_id="legacy",
        kind="custom_java",
        config={"implementation": "custom"},
    )

    assert _is_host_exact_authored_module(typed) is True
    shards = list(
        _module_shards(
            (typed, legacy),
            policy=ScalePolicy.from_environment(),
        )
    )
    assert len(shards) == 2
    typed_stage, typed_members = shards[0]
    assert typed_stage == "host"
    assert [member.module_id for member in typed_members] == ["typed"]

    node = _node(
        "typed-host",
        "generate:host",
        (),
        {
            "kind": "module-shard",
            "generation_stage": "host",
            "members": [
                {
                    "module_id": "typed",
                    "kind": "custom_java",
                    "config": {"typed_plan_ir": _plan()},
                }
            ],
        },
    )
    assert node.resource_class == "cpu_io"

