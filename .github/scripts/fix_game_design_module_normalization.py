from __future__ import annotations

from pathlib import Path

GAME_DESIGN = Path("minecraft_mod_ai/game_design.py")
TEST = Path("tests/test_game_design_router.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"repair anchor not found: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    source = GAME_DESIGN.read_text(encoding="utf-8")

    helper_anchor = '''\ndef _extract_valid_game_design(text: str) -> dict[str, Any]:\n'''
    helper = '''\ndef _first_nonempty_module_text(\n    value: dict[str, Any],\n    keys: Sequence[str],\n) -> str:\n    for key in keys:\n        candidate = value.get(key)\n        if isinstance(candidate, str) and candidate.strip():\n            return candidate.strip()\n    return ""\n\n\ndef _module_text_is_missing(value: dict[str, Any], key: str) -> bool:\n    if key not in value or value[key] is None:\n        return True\n    return isinstance(value[key], str) and not value[key].strip()\n\n\ndef _normalize_model_game_design(design: dict[str, Any]) -> dict[str, Any]:\n    """Canonicalize only recoverable module transport metadata.\n\n    The planner remains fail-closed for missing design fields, malformed collection\n    types, and non-object module entries. Local models sometimes preserve the module\n    selection itself while shortening its object to ``plugin_id`` or using harmless\n    aliases for the rationale. Those omissions are transport/schema noise rather than\n    missing game-design semantics, so fill only the canonical metadata for an already\n    model-authored module entry. Never add a module that the model did not emit.\n    """\n\n    normalized = dict(design)\n    modules = normalized.get("modules")\n    if not isinstance(modules, list):\n        return normalized\n\n    plugin_status = {\n        str(plugin.get("plugin_id", "")).strip(): str(plugin.get("status", "")).strip()\n        for plugin in _planner_plugin_manifest()["plugins"]\n        if str(plugin.get("plugin_id", "")).strip()\n        and str(plugin.get("status", "")).strip()\n    }\n    canonical_modules: list[Any] = []\n    for raw in modules:\n        if not isinstance(raw, dict):\n            canonical_modules.append(raw)\n            continue\n\n        module = dict(raw)\n        plugin_id = _first_nonempty_module_text(\n            module,\n            ("plugin_id", "module_id", "plugin", "id"),\n        )\n        if not plugin_id:\n            canonical_modules.append(module)\n            continue\n\n        if _module_text_is_missing(module, "plugin_id"):\n            module["plugin_id"] = plugin_id\n\n        if _module_text_is_missing(module, "status"):\n            module["status"] = plugin_status.get(plugin_id, "custom")\n\n        if _module_text_is_missing(module, "reason"):\n            reason = _first_nonempty_module_text(\n                module,\n                ("description", "purpose", "brief", "summary", "name", "label"),\n            )\n            # The identifier itself is the only safe deterministic fallback: it\n            # records the model-selected module without inventing a rationale.\n            module["reason"] = reason or plugin_id\n\n        canonical_modules.append(module)\n\n    normalized["modules"] = canonical_modules\n    return normalized\n\n\ndef _extract_valid_game_design(text: str) -> dict[str, Any]:\n'''
    source = replace_once(source, helper_anchor, helper, "module normalizer helper")

    extract_old = '''        nested = candidate.get("game_design")\n        possible = nested if isinstance(nested, dict) else candidate\n        if not isinstance(possible, dict):\n            continue\n        if set(possible) & set(_GAME_DESIGN_FIELDS):\n            _validate_design(possible)\n            return possible\n'''
    extract_new = '''        nested = candidate.get("game_design")\n        possible = nested if isinstance(nested, dict) else candidate\n        if not isinstance(possible, dict):\n            continue\n        if set(possible) & set(_GAME_DESIGN_FIELDS):\n            possible = _normalize_model_game_design(possible)\n            _validate_design(possible)\n            return possible\n'''
    source = replace_once(source, extract_old, extract_new, "extract normalization")

    standalone_old = '''        nested = candidate.get("game_design")\n        possible = nested if isinstance(nested, dict) else candidate\n        if not set(_GAME_DESIGN_FIELDS) <= set(possible):\n            continue\n        try:\n            _validate_design(possible)\n        except SpecValidationError:\n            continue\n        designs.append(possible)\n'''
    standalone_new = '''        nested = candidate.get("game_design")\n        possible = nested if isinstance(nested, dict) else candidate\n        if not set(_GAME_DESIGN_FIELDS) <= set(possible):\n            continue\n        possible = _normalize_model_game_design(possible)\n        try:\n            _validate_design(possible)\n        except SpecValidationError:\n            continue\n        designs.append(possible)\n'''
    source = replace_once(source, standalone_old, standalone_new, "standalone normalization")

    repair_old = '''    "title": "string", "pitch": "string", "core_loop": [], "progression": [],\n    "combat": {}, "mod_context": {}, "modules": [], "assets": [],\n    "acceptance_tests": []\n  }\n}\ncombat and mod_context must be JSON objects whose values, when present, are arrays of\nnon-empty strings. Use an empty object when the request has no relevant details; never\nreplace an array with a scalar string or a nested object.\n'''
    repair_new = '''    "title": "string", "pitch": "string", "core_loop": [], "progression": [],\n    "combat": {}, "mod_context": {},\n    "modules": [{"plugin_id":"from catalog or custom","status":"implemented|custom","reason":"why requested"}],\n    "assets": [{"id":"snake_case","kind":"item|block|entity|gui|environment","brief":"what to make"}],\n    "acceptance_tests": []\n  }\n}\nEvery modules entry must contain non-empty plugin_id, status, and reason strings. Every\nassets entry must contain non-empty id, kind, and brief strings. Use [] when there are\nno modules or assets. combat and mod_context must be JSON objects whose values, when\npresent, are arrays of non-empty strings. Use an empty object when the request has no\nrelevant details; never replace an array with a scalar string or a nested object.\n'''
    source = replace_once(source, repair_old, repair_new, "repair prompt nested schemas")

    compile(source, str(GAME_DESIGN), "exec")
    GAME_DESIGN.write_text(source, encoding="utf-8")

    test = TEST.read_text(encoding="utf-8")
    import_old = '''    _planner_plugin_manifest,\n    _repair_system_prompt,\n'''
    import_new = '''    _extract_valid_game_design,\n    _planner_plugin_manifest,\n    _repair_system_prompt,\n'''
    test = replace_once(test, import_old, import_new, "test import")

    regression = r'''


def test_multimodal_design_repairs_compact_module_metadata_without_third_model_call() -> None:
    """A local planner may keep the module choice while abbreviating its metadata."""

    payload = _valid_planner_payload()
    payload.pop("build_slice", None)
    payload["game_design"]["modules"] = [
        {"plugin_id": "custom_weather"},
        {
            "module_id": "seasonal_cooking",
            "description": "계절별 요리 진행 시스템",
        },
    ]
    router = _SequenceTextRouter(
        "not a JSON object",
        json.dumps(payload, ensure_ascii=False),
    )

    design, proposal = GameDesignPlanner(router).plan(
        "계절별 날씨와 요리 진행 시스템을 만들어줘."
    )

    assert len(router.calls) == 2
    assert design["modules"] == [
        {
            "plugin_id": "custom_weather",
            "status": "custom",
            "reason": "custom_weather",
        },
        {
            "module_id": "seasonal_cooking",
            "description": "계절별 요리 진행 시스템",
            "plugin_id": "seasonal_cooking",
            "status": "custom",
            "reason": "계절별 요리 진행 시스템",
        },
    ]
    assert proposal.spec.mod_id


def test_module_shape_recovery_does_not_accept_non_object_module_entries() -> None:
    payload = _valid_planner_payload()
    payload["game_design"]["modules"] = ["quest_system"]

    with pytest.raises(
        SpecValidationError,
        match="game_design.modules entries must contain",
    ):
        _extract_valid_game_design(json.dumps(payload))


def test_repair_prompt_restates_nested_module_and_asset_entry_contracts() -> None:
    prompt = _repair_system_prompt()
    assert '"plugin_id":"from catalog or custom"' in prompt
    assert '"status":"implemented|custom"' in prompt
    assert '"reason":"why requested"' in prompt
    assert '"id":"snake_case"' in prompt
    assert "Every modules entry must contain" in prompt
'''
    marker = "def test_game_design_prompts_define_strict_nested_collection_types() -> None:\n"
    if "test_multimodal_design_repairs_compact_module_metadata_without_third_model_call" not in test:
        if marker not in test:
            raise SystemExit("test insertion marker not found")
        test = test.replace(marker, regression + "\n\n" + marker, 1)
    compile(test, str(TEST), "exec")
    TEST.write_text(test, encoding="utf-8")


if __name__ == "__main__":
    main()
