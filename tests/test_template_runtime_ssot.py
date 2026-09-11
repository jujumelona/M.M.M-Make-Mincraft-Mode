from __future__ import annotations

import ast
import codecs
from pathlib import Path
from string import Formatter

import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "minecraft_mod_ai"
TEMPLATES = PKG / "templates"
SYSTEM_ROOT = TEMPLATES / "implementation" / "system_pack"

SYSTEM_FIELDS = {
    "persistent_store": {"package_name", "mod_id"},
    "config_loader": {"package_name", "mod_id"},
    "class_skill": {"package_name", "class_name", "resource"},
    "economy": {"package_name", "class_name", "resource"},
    "party_guild": {"package_name", "class_name", "resource"},
    "quest": {"package_name", "class_name", "resource"},
    "gui_networking": {"package_name", "mod_id", "class_name", "resource"},
}


def _render(name: str) -> str:
    raw = (SYSTEM_ROOT / f"{name}.java.fmt").read_text(encoding="utf-8")
    values = {
        "package_name": "dev.example.mod",
        "mod_id": "example_mod",
        "class_name": "ExampleSystem",
        "resource": "/data/example_mod/mmm_systems/example.json",
    }
    fields = {
        field_name for _, field_name, _, _ in Formatter().parse(raw)
        if field_name is not None
    }
    assert fields == SYSTEM_FIELDS[name]
    rendered = raw.format(**{key: values[key] for key in fields})
    return codecs.decode(rendered.encode("utf-8"), "unicode_escape")


def test_system_java_templates_are_external_resources() -> None:
    actual = {path.stem.removesuffix(".java") for path in SYSTEM_ROOT.glob("*.java.fmt")}
    assert actual == set(SYSTEM_FIELDS)
    for name in sorted(actual):
        source = _render(name)
        assert "package dev.example.mod.system;" in source
        assert "{{" not in source and "}}" not in source
        assert "{package_name}" not in source
        assert source.endswith("\n")
    assert 'path.contains("\\\\")' in _render("config_loader")


def test_python_system_template_modules_contain_no_java_source_bodies() -> None:
    for filename in (
        "system_templates_common.py",
        "system_templates_class_skill.py",
        "system_templates_economy.py",
        "system_templates_groups.py",
        "system_templates_quest.py",
        "system_templates_social.py",
    ):
        text = (PKG / filename).read_text(encoding="utf-8")
        assert "return f'''package" not in text
        assert "return f\"\"\"package" not in text
        assert "import net.minecraft" not in text
        ast.parse(text)
    assert "_party_java" not in (PKG / "system_templates_social.py").read_text(encoding="utf-8")


def test_record_execution_has_no_model_owned_status_loop() -> None:
    bounded = (PKG / "bounded_record_template.py").read_text(encoding="utf-8")
    batch = (PKG / "task_template_batch_runner.py").read_text(encoding="utf-8")
    assert "record/done" in bounded
    assert "applicability" in bounded
    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(bounded)))
    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(batch)))
    assert '"status"' not in batch


def test_feature_convergence_is_semantic_not_depth_limited() -> None:
    text = (PKG / "feature_template_pipeline.py").read_text(encoding="utf-8")
    assert "max_depth" not in text
    assert "semantic ancestry cycle/no-progress" in text
    assert "semantically duplicate children" in text


def test_workflow_sequences_have_one_authority() -> None:
    research = (PKG / "research_template_pipeline.py").read_text(encoding="utf-8")
    reuse = (PKG / "reuse_template_pipeline.py").read_text(encoding="utf-8")
    assert "RESEARCH_SEQUENCE" not in research
    assert "REUSE_SEQUENCE" not in reuse
    assert 'load_template("research/workflow")' in research
    assert 'load_template("reuse/workflow")' in reuse


def test_prompt_policy_is_shared_and_capture_is_removed() -> None:
    assert not (TEMPLATES / "prompt" / "capture.yaml").exists()
    workflow = yaml.safe_load((TEMPLATES / "prompt" / "workflow.yaml").read_text(encoding="utf-8"))
    assert "prompt/capture" not in workflow["steps"]
    policy = yaml.safe_load((TEMPLATES / "prompt" / "policy.yaml").read_text(encoding="utf-8"))
    common = set(policy["rules"])
    for identifier in workflow["steps"]:
        path = TEMPLATES / (identifier + ".yaml")
        child = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert common.isdisjoint(set(child.get("rules", ())))


def test_response_contracts_are_not_python_hardcoded() -> None:
    text = (PKG / "model_response_templates.py").read_text(encoding="utf-8")
    assert "_TEMPLATES" not in text
    assert "response/contracts.json" in text
    assert (TEMPLATES / "response" / "contracts.json").is_file()


def test_package_data_contains_all_template_resource_types() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"templates/**/*.yaml"' in pyproject
    assert '"templates/**/*.json"' in pyproject
    assert '"templates/**/*.java.fmt"' in pyproject
