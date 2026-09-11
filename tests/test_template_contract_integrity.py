from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "minecraft_mod_ai" / "templates"
LEGACY_MIRROR_ROOT = REPO_ROOT / "templates"
PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
FORBIDDEN_SENTINELS = {"not_applicable", "blocked"}


def _yaml_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.yaml"))


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _enum_rejects_blank_string(spec: dict[str, Any]) -> bool:
    enum = spec.get("enum")
    if not isinstance(enum, list) or not enum:
        return False
    string_values = [value for value in enum if isinstance(value, str)]
    return bool(string_values) and all(value.strip() for value in string_values)


def _assert_schema_required_strings_are_nonempty(node: Any, path: Path) -> None:
    if isinstance(node, dict):
        properties = node.get("properties")
        required = node.get("required")
        if isinstance(properties, dict) and isinstance(required, list):
            for name in required:
                spec = properties.get(name)
                if not isinstance(spec, dict):
                    continue
                type_spec = spec.get("type")
                string_typed = type_spec == "string" or (
                    isinstance(type_spec, list) and "string" in type_spec
                )
                nullable = isinstance(type_spec, list) and "null" in type_spec
                if string_typed and not nullable and not _enum_rejects_blank_string(spec):
                    assert spec.get("minLength", 0) >= 1, (
                        f"{path}: required string property {name!r} must reject empty strings"
                    )
        for child in node.values():
            _assert_schema_required_strings_are_nonempty(child, path)
    elif isinstance(node, list):
        for child in node:
            _assert_schema_required_strings_are_nonempty(child, path)


def _assert_no_null_contract_values(data: dict[str, Any], path: Path) -> None:
    if "translation" not in path.parts:
        return
    for section_name in ("input", "task", "output"):
        section = data.get(section_name)
        if section is None:
            continue
        for value in _walk(section):
            assert value is not None, (
                f"{path}: {section_name} contains an untyped YAML null contract value"
            )


def _assert_rule_uniqueness(data: dict[str, Any], path: Path) -> None:
    rules = data.get("rules")
    if not isinstance(rules, list):
        return
    normalized = [" ".join(str(rule).split()).casefold() for rule in rules]
    assert len(normalized) == len(set(normalized)), f"{path}: duplicate rules"


def _assert_placeholders_declared(data: dict[str, Any], path: Path) -> None:
    render = data.get("render")
    if not isinstance(render, dict):
        return
    body = render.get("body")
    if not isinstance(body, str):
        return
    placeholders = set(PLACEHOLDER_RE.findall(body))
    if not placeholders:
        return
    declared: set[str] = set()
    for input_key in ("inputs", "input"):
        section = data.get(input_key)
        if isinstance(section, dict):
            declared.update(str(name) for name in section)
    missing = placeholders - declared
    assert not missing, f"{path}: undeclared render placeholders: {sorted(missing)}"


def _assert_no_forbidden_schema_sentinels(data: Any, path: Path) -> None:
    def visit(node: Any, in_contract: bool = False) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_contract = in_contract or key in {
                    "record_schema",
                    "input_schema",
                    "output_schema",
                    "schema",
                }
                visit(value, child_contract)
        elif isinstance(node, list):
            for value in node:
                visit(value, in_contract)
        elif in_contract and isinstance(node, str):
            assert node.casefold() not in FORBIDDEN_SENTINELS, (
                f"{path}: forbidden sentinel {node!r} remains in a contract"
            )

    visit(data)


def test_all_canonical_templates_are_valid_and_contract_safe() -> None:
    files = _yaml_files(CANONICAL_ROOT)
    assert files, "canonical template tree is empty"
    for path in files:
        data = _load(path)
        assert isinstance(data, dict), f"{path}: template root must be a mapping"

        template_id = data.get("id")
        if template_id is not None:
            expected_id = path.relative_to(CANONICAL_ROOT).with_suffix("").as_posix()
            assert template_id == expected_id, (
                f"{path}: id {template_id!r} does not match path {expected_id!r}"
            )

        _assert_schema_required_strings_are_nonempty(data, path)
        _assert_no_null_contract_values(data, path)
        _assert_rule_uniqueness(data, path)
        _assert_placeholders_declared(data, path)
        _assert_no_forbidden_schema_sentinels(data, path)


def test_legacy_root_templates_cannot_drift_from_runtime_ssot() -> None:
    if not LEGACY_MIRROR_ROOT.exists():
        return
    for mirror in _yaml_files(LEGACY_MIRROR_ROOT):
        relative = mirror.relative_to(LEGACY_MIRROR_ROOT)
        canonical = CANONICAL_ROOT / relative
        assert canonical.exists(), f"legacy mirror has no canonical template: {relative}"
        assert mirror.read_bytes() == canonical.read_bytes(), (
            f"legacy template drift: {relative}; edit minecraft_mod_ai/templates first"
        )
