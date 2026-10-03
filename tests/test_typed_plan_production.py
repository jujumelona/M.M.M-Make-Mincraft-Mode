from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.typed_plan_production import generate_typed_plan_module


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

    receipt = generate_typed_plan_module(root, module=module)

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
