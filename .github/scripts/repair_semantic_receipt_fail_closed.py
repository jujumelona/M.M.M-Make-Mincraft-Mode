from pathlib import Path

path = Path("minecraft_mod_ai/complete_orchestrator.py")
text = path.read_text(encoding="utf-8")
old = '''def _semantic_execution_observations(
    members: Iterable[ProductionModule],
    receipts: Iterable[dict[str, Any]],
    *,
    downstream_ids: Callable[[str], Iterable[str]],
) -> list[dict[str, Any]]:
    """Attribute receipts by explicit ownership, never by list position."""
    member_by_id = {module.module_id: module for module in members}
    observations: list[dict[str, Any]] = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        owner_ids = _receipt_owned_module_ids(receipt)
        if not owner_ids and len(member_by_id) == 1:
            owner_ids = tuple(member_by_id)
        for module_id in owner_ids:
            module = member_by_id.get(module_id)
            if module is None:
                continue
            observation = _semantic_execution_observation(
                module,
                receipt,
                dependent_ids=downstream_ids(module_id),
            )
            if observation is not None:
                observations.append(observation)
    return observations
'''
new = '''def _semantic_execution_observations(
    members: Iterable[ProductionModule],
    receipts: Iterable[dict[str, Any]],
    *,
    downstream_ids: Callable[[str], Iterable[str]],
) -> list[dict[str, Any]]:
    """Attribute receipts by explicit ownership and fail closed on evidence gaps."""
    member_by_id = {module.module_id: module for module in members}
    member_ids = set(member_by_id)
    tracked_ids = {
        module_id
        for module_id, module in member_by_id.items()
        if isinstance(module.config, dict)
        and isinstance(module.config.get("evidence_task"), dict)
    }
    observations: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        owner_ids = _receipt_owned_module_ids(receipt)
        if not owner_ids:
            if len(member_by_id) == 1:
                owner_ids = tuple(member_by_id)
            elif tracked_ids:
                raise CompleteProductionError(
                    "SEMANTIC_RECEIPT_OWNERSHIP_MISSING: "
                    "multi-module generation receipt did not declare module ownership"
                )
            else:
                continue
        foreign_ids = sorted(set(owner_ids) - member_ids)
        if foreign_ids:
            raise CompleteProductionError(
                "SEMANTIC_RECEIPT_OWNER_OUTSIDE_NODE: " + ", ".join(foreign_ids)
            )
        for module_id in owner_ids:
            module = member_by_id[module_id]
            observation = _semantic_execution_observation(
                module,
                receipt,
                dependent_ids=downstream_ids(module_id),
            )
            if observation is not None:
                observations.append(observation)
                observed_ids.add(module_id)
    missing_ids = sorted(tracked_ids - observed_ids)
    if missing_ids:
        raise CompleteProductionError(
            "SEMANTIC_RECEIPT_COVERAGE_MISSING: " + ", ".join(missing_ids)
        )
    return observations
'''
if old not in text:
    if new in text:
        raise SystemExit(0)
    raise SystemExit("semantic observation function marker not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
