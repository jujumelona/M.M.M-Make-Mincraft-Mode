from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"{label}: patch anchor missing")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "minecraft_mod_ai/planir_mutation_authority_contract.py",
        '''            and str(context.evidence_source or "") == "evidence_fresh_owned_anchor"\n''',
        '''            and str(context.evidence_source or "") in {\n                "evidence_fresh_owned_anchor",\n                "evidence_host_reserved_owned_anchor",\n            }\n''',
        "host-reserved target pin",
    )

    replace_once(
        "minecraft_mod_ai/progress_aware_tool_loop.py",
        '''    pinned = _canonical_mutation_path(context.target_path)\n    operation = str(arguments.get("operation", "")).strip().casefold()\n    if pinned and not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:\n        return (\n            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "\n            f"{pinned!r} cannot be recreated by {operation!r}."\n        )\n    if not context.is_mutation_ready:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "\n            "repository-localized target context."\n        )\n\n    supplied = ""\n    for key in _SOURCE_EDIT_PATH_KEYS:\n        value = arguments.get(key)\n        if isinstance(value, str) and value.strip():\n            supplied = _canonical_mutation_path(value)\n            break\n\n    if not pinned or not supplied:\n''',
        '''    pinned = _canonical_mutation_path(context.target_path)\n    supplied = ""\n    for key in _SOURCE_EDIT_PATH_KEYS:\n        value = arguments.get(key)\n        if isinstance(value, str) and value.strip():\n            supplied = _canonical_mutation_path(value)\n            break\n\n    if not pinned or not supplied:\n''',
        "drift-before-creation prefix",
    )

    replace_once(
        "minecraft_mod_ai/progress_aware_tool_loop.py",
        '''    if supplied != pinned:\n        return (\n            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "\n            f"apply_source_edit requested {supplied!r}."\n        )\n\n    return None\n''',
        '''    if supplied != pinned:\n        return (\n            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "\n            f"apply_source_edit requested {supplied!r}."\n        )\n\n    operation = str(arguments.get("operation", "")).strip().casefold()\n    if not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:\n        return (\n            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "\n            f"{pinned!r} cannot be recreated by {operation!r}."\n        )\n    if not context.is_mutation_ready:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "\n            "repository-localized target context."\n        )\n\n    return None\n''',
        "drift-before-creation suffix",
    )


if __name__ == "__main__":
    main()
