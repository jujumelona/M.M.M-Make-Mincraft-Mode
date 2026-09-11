from __future__ import annotations

from pathlib import Path


def replace_or_verify(path: str, old: str, new: str, label: str) -> None:
    target = Path(path)
    text = target.read_text(encoding='utf-8')
    if new in text:
        return
    if old not in text:
        raise SystemExit(f'{label}: patch anchor missing')
    target.write_text(text.replace(old, new, 1), encoding='utf-8')


def main() -> None:
    replace_or_verify(
        'minecraft_mod_ai/progress_aware_tool_loop.py',
        '''    if tool_name != "apply_source_edit":
        return None
    if context is None or not context.is_mutation_ready:
        return (
            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "
            "repository-localized target context."
        )

    pinned = _canonical_mutation_path(context.target_path)
    supplied = ""
''',
        '''    if tool_name != "apply_source_edit":
        return None
    if context is None:
        return (
            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "
            "repository-localized target context."
        )

    pinned = _canonical_mutation_path(context.target_path)
    operation = str(arguments.get("operation", "")).strip().casefold()
    if pinned and not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:
        return (
            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "
            f"{pinned!r} cannot be recreated by {operation!r}."
        )
    if not context.is_mutation_ready:
        return (
            "MUTATION_TARGET_UNBOUND: apply_source_edit requires a READY "
            "repository-localized target context."
        )

    supplied = ""
''',
        'creation conflict priority',
    )
    path = Path('minecraft_mod_ai/progress_aware_tool_loop.py')
    text = path.read_text(encoding='utf-8')
    redundant = '''    operation = str(arguments.get("operation", "")).strip().casefold()
    if not context.is_new_file and operation in _SOURCE_CREATE_OPERATIONS:
        return (
            "MUTATION_TARGET_CREATION_CONFLICT: existing localized target "
            f"{pinned!r} cannot be recreated by {operation!r}."
        )
    return None
'''
    if redundant in text:
        text = text.replace(redundant, '    return None\n', 1)
        path.write_text(text, encoding='utf-8')

    replace_or_verify(
        'minecraft_mod_ai/small_model_task_capsule_contract.py',
        '''    unsupported = [anchor for anchor in anchors if not target_is_writable(anchor.status)]
    if unsupported:
        rendered = [f"{anchor.locator}:{anchor.status or '<empty>'}" for anchor in unsupported]
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_TARGET_STATUS_INVALID: owned workspace targets use unsupported status "
            f"{rendered!r}."
        )

    bindings = _matching_bindings(task, task_id)
''',
        '''    unsupported = [anchor for anchor in anchors if not target_is_writable(anchor.status)]
    if unsupported:
        provisional_bindings = _matching_bindings(task, task_id)
        provisional_candidates = _binding_symbol_candidates(provisional_bindings)
        if len(provisional_candidates) == 1:
            provisional_primary_path, _ = provisional_candidates[0]
            provisional_primary = next(
                (anchor for anchor in unsupported if anchor.path == provisional_primary_path),
                None,
            )
            if provisional_primary is not None:
                raise TaskCapsuleContractError(
                    "TASK_CAPSULE_PRIMARY_NOT_RESERVED: planned custom-Java primary target has "
                    f"unsupported status {provisional_primary.status!r}."
                )
        rendered = [f"{anchor.locator}:{anchor.status or '<empty>'}" for anchor in unsupported]
        raise TaskCapsuleContractError(
            "TASK_CAPSULE_TARGET_STATUS_INVALID: owned workspace targets use unsupported status "
            f"{rendered!r}."
        )

    bindings = _matching_bindings(task, task_id)
''',
        'primary status classification',
    )

    replace_or_verify(
        'tests/test_complete_production.py',
        "result = generate_geckolib_entity_assets(project_root=project, mod_id='complete_test', package_name='ai.minecraft.complete_test', entity_id=entity_id)",
        "result = generate_geckolib_entity_assets(project_root=project, mod_id='complete_test', package_name='ai.minecraft.complete_test', entity_id=entity_id, spawn_group='monster', texture_color='#5ba6d8')",
        'geckolib explicit design fixture',
    )
    replace_or_verify(
        'tests/test_complete_production.py',
        "ProductionModule('frost_guard', 'entity', {'max_health': 60})",
        "ProductionModule('frost_guard', 'entity', {'max_health': 60, 'attack_damage': 8, 'movement_speed': 0.27, 'follow_range': 40, 'archetype': 'biped', 'behavior': 'hostile_melee', 'entity_width': 0.8, 'entity_height': 2.0, 'spawn_group': 'monster', 'main_color': '#5ba6d8'})",
        'orchestrator explicit entity design fixture',
    )


if __name__ == '__main__':
    main()
