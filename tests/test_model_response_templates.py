"""Prevent response schema omissions and verify executable patch compatibility."""
import ast
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.model_response_templates import response_schema, response_template_prompt
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher, sha256_bytes


@pytest.mark.parametrize("name", ["repair", "visual_review", "capabilities", "coder_summary"])
def test_response_prompt_uses_exact_decode_schema(name):
    schema = response_schema(name)
    Draft202012Validator.check_schema(schema)
    assert json.loads(response_template_prompt(name).split(": ", 1)[1]) == schema


def test_repair_template_accepts_patch_that_applies_with_hash_guard(tmp_path):
    path = tmp_path / "src/main/java/Test.java"
    path.parent.mkdir(parents=True)
    path.write_text("class Test { int value = 1; }")
    patch = {"operations": [{
        "operation": "edit", "path": "src/main/java/Test.java",
        "expected_sha256": sha256_bytes(path.read_bytes()),
        "replacements": [{"old": "value = 1", "new": "value = 2", "count": 1}],
    }]}
    Draft202012Validator(response_schema("repair")).validate(patch)
    TransactionalSourcePatcher(tmp_path).apply(patch["operations"])
    assert "value = 2" in path.read_text()


def test_every_explicit_json_generation_call_supplies_a_schema():
    root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    missing = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name != "generate_text":
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            fmt = kwargs.get("response_format")
            if isinstance(fmt, ast.Constant) and fmt.value == "json":
                if "response_schema" not in kwargs:
                    missing.append(f"{path.name}:{node.lineno}")
    assert missing == []
