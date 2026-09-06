from __future__ import annotations

import json
from pathlib import Path

import minecraft_mod_ai
from minecraft_mod_ai.agent_capability_context import build_agent_capability_context
from minecraft_mod_ai.contract_schema_catalog import contract_schema_manifest


def _canonical_contract_paths() -> set[str]:
    package_root = Path(minecraft_mod_ai.__file__).resolve().parent
    return {
        path.relative_to(package_root.parent).as_posix()
        for path in package_root.rglob("*.py")
        if path.name == "target_contract.py"
        or path.name.endswith(("_contract.py", "_contracts.py"))
    }


def _decode_context(text: str) -> dict[str, object]:
    prefix = "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n"
    assert text.startswith(prefix)
    return json.loads(text[len(prefix) :])


def test_catalog_covers_every_canonical_contract_file() -> None:
    manifest = contract_schema_manifest()
    represented = {str(item["path"]) for item in manifest}
    assert represented == _canonical_contract_paths()
    assert all("parse_error" not in item for item in manifest)


def test_catalog_exposes_target_contract_field_shape() -> None:
    manifest = contract_schema_manifest()
    target_module = next(
        item for item in manifest if item["path"] == "minecraft_mod_ai/target_contract.py"
    )
    target_type = next(item for item in target_module["types"] if item["name"] == "TargetContract")
    fields = {field["name"]: field["type"] for field in target_type["fields"]}

    assert fields["minecraft_version"] == "str"
    assert fields["loader"] == "str"
    assert fields["resource_pack_format"] == "int"
    assert fields["deterministic_module_kinds"] == "frozenset[str]"


def test_small_agent_context_contains_the_same_contract_manifest() -> None:
    context = _decode_context(build_agent_capability_context("research", ()))
    assert context["schema_version"] == "mmm/agent-capability-context-v6"
    assert context["type_contracts"] == json.loads(json.dumps(contract_schema_manifest()))
    assert "obey type_contracts exactly" in str(context["routing_policy"])
