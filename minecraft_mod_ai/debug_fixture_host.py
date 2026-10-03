from __future__ import annotations

"""Deterministic host compiler for the internal Debug fixture.

Debug Mode exists to exercise the real build/verification pipeline without depending
on planner/coder model output. The source below is projected only from the immutable
HOST target/template authority used by normal generation grounding.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .complete_spec import ProductionModule
from .generation_implementation_grounding import (
    build_generation_implementation_grounding,
)


class DebugFixtureHostError(RuntimeError):
    pass


def _debug_grounding(*, minecraft_version: str) -> dict[str, Any]:
    module = ProductionModule(
        module_id="debug_token",
        kind="item",
        config={
            "semantic_kind": "item",
            "implementation_responsibilities": ["registry"],
        },
    )
    grounding = build_generation_implementation_grounding(
        module,
        minecraft_version=minecraft_version,
    )
    if not isinstance(grounding, dict):
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_HOST_GROUNDING_MISSING: item registry authority is unavailable"
        )
    facts = grounding.get("facts")
    if not isinstance(facts, list) or len(facts) != 1 or not isinstance(facts[0], dict):
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_HOST_GROUNDING_INVALID: expected one item registry fact"
        )
    return grounding


def debug_fixture_source_contract(
    *,
    package_name: str,
    minecraft_version: str,
) -> dict[str, Any]:
    grounding = _debug_grounding(minecraft_version=minecraft_version)
    fact = grounding["facts"][0]
    symbols = fact.get("api_symbols")
    if not isinstance(symbols, Mapping):
        raise DebugFixtureHostError("DEBUG_FIXTURE_HOST_SYMBOLS_MISSING")

    required_keys = [
        key
        for key in ("resource_key_create", "register_item", "item_set_id")
        if isinstance(symbols.get(key), Mapping)
    ]
    if "resource_key_create" not in required_keys or "register_item" not in required_keys:
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_HOST_REGISTRY_SYMBOLS_INCOMPLETE"
        )
    required_specs = {
        key: dict(symbols[key])
        for key in required_keys
        if isinstance(symbols.get(key), Mapping)
    }
    return {
        "schema_version": "mmm/debug-source-contract-v1",
        "path": (
            "src/main/java/"
            + package_name.replace(".", "/")
            + "/DebugToken.java"
        ),
        "identifier": "debug_token",
        "binding_field": "DEBUG_TOKEN",
        "required_host_symbol_keys": required_keys,
        "required_host_symbol_specs": required_specs,
        "forbidden_lifecycle_symbols": [
            "ModInitializer",
            "ClientModInitializer",
            "onInitialize",
            "onInitializeClient",
        ],
        "grounding_sha256": grounding.get("grounding_sha256", ""),
    }


def _selected_templates(fact: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    templates = fact.get("templates")
    if not isinstance(templates, list):
        raise DebugFixtureHostError("DEBUG_FIXTURE_HOST_TEMPLATES_MISSING")

    key_template: Mapping[str, Any] | None = None
    register_template: Mapping[str, Any] | None = None
    for raw in templates:
        if not isinstance(raw, Mapping):
            continue
        usage = raw.get("symbol_usage")
        names = {
            str(value).strip()
            for value in usage
            if str(value).strip()
        } if isinstance(usage, list) else set()
        if "resource_key_create" in names:
            key_template = raw
        if "register_item" in names:
            register_template = raw

    if key_template is None or register_template is None:
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_HOST_TEMPLATE_TOPOLOGY_INCOMPLETE"
        )
    return key_template, register_template


def render_debug_fixture_source(
    *,
    package_name: str,
    mod_id: str,
    minecraft_version: str,
) -> str:
    grounding = _debug_grounding(minecraft_version=minecraft_version)
    fact = grounding["facts"][0]
    key_template, register_template = _selected_templates(fact)

    key_body = str(key_template.get("render_body") or "").strip()
    register_body = str(register_template.get("render_body") or "").strip()
    if not key_body or not register_body:
        raise DebugFixtureHostError("DEBUG_FIXTURE_HOST_TEMPLATE_BODY_EMPTY")

    replacements = {
        "{{java_constant}}": "DEBUG_TOKEN",
        "{{mod_id}}": mod_id,
        "{{registry_path}}": "debug_token",
        "ModItemIds.DEBUG_TOKEN_KEY": "DEBUG_TOKEN_KEY",
    }
    for before, after in replacements.items():
        key_body = key_body.replace(before, after)
        register_body = register_body.replace(before, after)

    unresolved = [
        token
        for token in ("{{java_constant}}", "{{mod_id}}", "{{registry_path}}")
        if token in key_body or token in register_body
    ]
    if unresolved:
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_HOST_TEMPLATE_PLACEHOLDER_UNRESOLVED: "
            + ", ".join(unresolved)
        )

    imports = fact.get("required_imports")
    owners = []
    if isinstance(imports, list):
        owners = list(
            dict.fromkeys(
                str(owner).strip()
                for owner in imports
                if str(owner).strip()
            )
        )

    body_lines: list[str] = []
    for fragment in (key_body, register_body):
        if body_lines:
            body_lines.append("")
        body_lines.extend(fragment.splitlines())

    return "\n".join(
        [
            f"package {package_name};",
            "",
            *(f"import {owner};" for owner in owners),
            "",
            "public final class DebugToken {",
            "    private DebugToken() {}",
            "",
            *("    " + line if line else "" for line in body_lines),
            "}",
            "",
        ]
    )


def materialize_debug_fixture_source(
    project_root: str | Path,
    *,
    package_name: str,
    mod_id: str,
    minecraft_version: str,
    source_contract: Mapping[str, Any],
) -> Path:
    relative = str(source_contract.get("path") or "").strip()
    expected = (
        "src/main/java/"
        + package_name.replace(".", "/")
        + "/DebugToken.java"
    )
    if relative != expected:
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_SOURCE_CONTRACT_PATH_MISMATCH: "
            f"expected={expected!r}, actual={relative!r}"
        )

    root = Path(project_root).expanduser().resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise DebugFixtureHostError(
            "DEBUG_FIXTURE_SOURCE_CONTRACT_ESCAPED_PROJECT"
        ) from exc
    if target.exists() and (not target.is_file() or target.is_symlink()):
        raise DebugFixtureHostError("DEBUG_FIXTURE_SOURCE_TARGET_UNSAFE")

    source = render_debug_fixture_source(
        package_name=package_name,
        mod_id=mod_id,
        minecraft_version=minecraft_version,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8", newline="\n")
    return target


__all__ = [
    "DebugFixtureHostError",
    "debug_fixture_source_contract",
    "materialize_debug_fixture_source",
    "render_debug_fixture_source",
]
