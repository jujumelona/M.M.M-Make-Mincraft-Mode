from __future__ import annotations

import runpy
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
    # Keep the canonical acceptance alias source-form required by the SSOT contract,
    # while making it an intentional runtime dependency rather than a lint-only import.
    replace_once(
        "minecraft_mod_ai/evidence_first_planning.py",
        "from .acceptance_contracts import is_public_acceptance as _is_public_acceptance\n\n",
        "from .acceptance_contracts import is_public_acceptance as _is_public_acceptance\n\nassert callable(_is_public_acceptance)\n\n",
        "acceptance alias lint",
    )

    # Resolve mutation authority in strict order: bind path, reject drift, then
    # classify same-target creation conflict, then require a READY source body.
    replace_once(
        "minecraft_mod_ai/progress_aware_tool_loop.py",
        '''    pinned = _canonical_mutation_path(context.target_path)\n    operation = str(arguments.get("operation", "")).strip().casefold()\n    if pinned and not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:\n        return (\n            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "\n            f"{pinned!r} cannot be recreated by {operation!r}."\n        )\n    if not context.is_mutation_ready:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "\n            "repository-localized target context."\n        )\n\n    supplied = ""\n    for key in _SOURCE_EDIT_PATH_KEYS:\n        value = arguments.get(key)\n        if isinstance(value, str) and value.strip():\n            supplied = _canonical_mutation_path(value)\n            break\n\n    if not pinned or not supplied:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires the pinned target "\n            "path in its model payload."\n        )\n    if supplied != pinned:\n        return (\n            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "\n            f"apply_source_edit requested {supplied!r}."\n        )\n\n    return None\n''',
        '''    pinned = _canonical_mutation_path(context.target_path)\n    supplied = ""\n    for key in _SOURCE_EDIT_PATH_KEYS:\n        value = arguments.get(key)\n        if isinstance(value, str) and value.strip():\n            supplied = _canonical_mutation_path(value)\n            break\n\n    if not pinned or not supplied:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires the pinned target "\n            "path in its model payload."\n        )\n    if supplied != pinned:\n        return (\n            f"MUTATION_TARGET_DRIFT: pinned target {pinned!r} but "\n            f"apply_source_edit requested {supplied!r}."\n        )\n\n    operation = str(arguments.get("operation", "")).strip().casefold()\n    if not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:\n        return (\n            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "\n            f"{pinned!r} cannot be recreated by {operation!r}."\n        )\n    if not context.is_mutation_ready:\n        return (\n            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "\n            "repository-localized target context."\n        )\n\n    return None\n''',
        "mutation authority ordering",
    )

    # Promote the canonical extended-content catalog iterator to a single
    # source-validation gate. The staging script is deterministic/idempotent.
    staging = Path("tools/_apply_extended_catalog_validation_gate.py")
    if staging.is_file():
        runpy.run_path(str(staging), run_name="__main__")


if __name__ == "__main__":
    main()
