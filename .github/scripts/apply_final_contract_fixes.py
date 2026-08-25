from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one anchor, found {count}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_scalar_protocol() -> None:
    path = Path("minecraft_mod_ai/source_edit_scalar_protocol_contract.py")
    replace_once(
        path,
        '_ACCEPTED_OPERATIONS = tuple(sorted(set(_CANONICAL_OPERATIONS) | set(_OPERATION_ALIASES.keys())))\n',
        '_MODEL_OPERATION_ENUM = (*_CANONICAL_OPERATIONS, "replace", "create", "delete")\n'
        'SOURCE_EDIT_PARAMETER_ALIASES = {\n'
        '    "file": "path",\n'
        '    "target_path": "path",\n'
        '    "target_file": "path",\n'
        '    "new_text": "new",\n'
        '    "new_content": "new",\n'
        '    "replacement": "new",\n'
        '    "old_text": "old",\n'
        '    "code": "content",\n'
        '    "body": "content",\n'
        '}\n',
    )
    replace_once(path, '"enum": list(_ACCEPTED_OPERATIONS),', '"enum": list(_MODEL_OPERATION_ENUM),')

    text = path.read_text(encoding="utf-8")
    start = text.index("def _canonicalize_payload_aliases(")
    end = text.index("\n\ndef _normalize_operation", start)
    replacement = (
        'def _canonicalize_payload_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:\n'
        '    normalized = dict(payload)\n'
        '    for alias, canonical in SOURCE_EDIT_PARAMETER_ALIASES.items():\n'
        '        if alias in normalized and canonical not in normalized:\n'
        '            normalized[canonical] = normalized.pop(alias)\n'
        '    return normalized\n'
    )
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    replace_once(
        path,
        '__all__ = ["SOURCE_EDIT_SCHEMA", "materialize_model_source_edit"]',
        '__all__ = ["SOURCE_EDIT_PARAMETER_ALIASES", "SOURCE_EDIT_SCHEMA", "materialize_model_source_edit"]',
    )


def patch_qwen_parser() -> None:
    path = Path("minecraft_mod_ai/model_adapters/qwen_tool_parser.py")
    replace_once(
        path,
        'from ..model_tool_aliases import resolve_exposed_model_tool\n',
        'from ..model_tool_aliases import resolve_exposed_model_tool\n'
        'from ..source_edit_scalar_protocol_contract import SOURCE_EDIT_PARAMETER_ALIASES\n',
    )
    replace_once(
        path,
        '_MAX_CONTAINER_DEPTH = 3\n',
        '_APPLY_SOURCE_EDIT_ALIASES = {\n'
        '    **SOURCE_EDIT_PARAMETER_ALIASES,\n'
        '    **_APPLY_SOURCE_EDIT_TRANSPORT_ALIASES,\n'
        '}\n'
        '_MAX_CONTAINER_DEPTH = 3\n',
    )
    replace_once(
        path,
        '_APPLY_SOURCE_EDIT_TRANSPORT_ALIASES.get(emitted_key)',
        '_APPLY_SOURCE_EDIT_ALIASES.get(emitted_key)',
    )
    replace_once(
        path,
        'for alias, canonical in _APPLY_SOURCE_EDIT_TRANSPORT_ALIASES.items()',
        'for alias, canonical in _APPLY_SOURCE_EDIT_ALIASES.items()',
    )


def patch_java_context_normalization() -> None:
    path = Path("minecraft_mod_ai/trajectory_memory.py")
    marker = "\n\ndef _collect_execution_context("
    text = path.read_text(encoding="utf-8")
    if text.count(marker) != 1:
        raise SystemExit("trajectory_memory.py: unexpected _collect_execution_context anchor count")
    helper = (
        '\n\ndef _normalize_execution_context_value(\n'
        '    key: str,\n'
        '    value: Any,\n'
        ') -> str | int | float | bool | None:\n'
        '    normalized = _normalize_context_value(value)\n'
        '    if key != "java_version" or normalized is None or isinstance(normalized, bool):\n'
        '        return normalized\n'
        '    if isinstance(normalized, str) and normalized.isdecimal():\n'
        '        return int(normalized)\n'
        '    if isinstance(normalized, float) and normalized.is_integer():\n'
        '        return int(normalized)\n'
        '    return normalized\n'
    )
    path.write_text(text.replace(marker, helper + marker, 1), encoding="utf-8")
    replace_once(
        path,
        '                normalized = _normalize_context_value(raw_value)\n',
        '                normalized = _normalize_execution_context_value(key, raw_value)\n',
    )


def patch_regression_contracts() -> None:
    path = Path("tests/test_temporary_skill_reuse.py")
    replace_once(
        path,
        '    monkeypatch.setenv("MMM_LOADER", "fabric")\n    messages = [\n',
        '    monkeypatch.setenv("MMM_LOADER", "fabric")\n'
        '    monkeypatch.setenv("MMM_JAVA_VERSION", "21")\n'
        '    messages = [\n',
    )

    path = Path("tests/test_source_edit_alias_ownership.py")
    replace_once(
        path,
        'from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA\n',
        'from minecraft_mod_ai.source_edit_scalar_protocol_contract import (\n'
        '    SOURCE_EDIT_PARAMETER_ALIASES,\n'
        '    SOURCE_EDIT_SCHEMA,\n'
        ')\n',
    )
    old = (
        '    scalar_aliases = {\n'
        '        "file",\n'
        '        "target_path",\n'
        '        "target_file",\n'
        '        "new_text",\n'
        '        "new_content",\n'
        '        "replacement",\n'
        '        "old_text",\n'
        '        "code",\n'
        '        "body",\n'
        '    }\n'
        '    assert transport.isdisjoint(properties)\n'
        '    assert scalar_aliases <= properties\n'
        '    assert scalar_aliases.isdisjoint(transport)\n'
    )
    new = (
        '    scalar_aliases = set(SOURCE_EDIT_PARAMETER_ALIASES)\n'
        '    assert transport.isdisjoint(properties)\n'
        '    assert scalar_aliases <= properties\n'
        '    assert scalar_aliases.isdisjoint(transport)\n'
        '    assert all(\n'
        '        qwen_tool_parser._APPLY_SOURCE_EDIT_ALIASES[alias] == canonical\n'
        '        for alias, canonical in SOURCE_EDIT_PARAMETER_ALIASES.items()\n'
        '    )\n'
    )
    replace_once(path, old, new)


def main() -> None:
    patch_scalar_protocol()
    patch_qwen_parser()
    patch_java_context_normalization()
    patch_regression_contracts()


if __name__ == "__main__":
    main()
