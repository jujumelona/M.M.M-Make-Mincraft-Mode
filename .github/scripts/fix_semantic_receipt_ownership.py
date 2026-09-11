from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old in text:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if new not in text:
        raise SystemExit(f"repair marker not found: {path}")


path = Path("minecraft_mod_ai/complete_orchestrator.py")
text = path.read_text(encoding="utf-8")
marker = "\n@dataclass(frozen=True)\nclass CompleteExecutionOptions:"
helper = '''


def _receipt_owned_module_ids(receipt: dict[str, Any]) -> tuple[str, ...]:
    """Return only module ownership explicitly declared by a generation receipt."""
    owned: set[str] = set()
    for key in ("module_ids", "modules"):
        values = receipt.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if isinstance(value, str) and value.strip():
                owned.add(value.strip())
            elif isinstance(value, dict):
                module_id = value.get("module_id")
                if isinstance(module_id, str) and module_id.strip():
                    owned.add(module_id.strip())
    for key in ("module_id", "entity_id"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            owned.add(value.strip())
    return tuple(sorted(owned))


def _semantic_execution_observations(
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
if helper.strip() not in text:
    if marker not in text:
        raise SystemExit("complete_orchestrator insertion marker not found")
    text = text.replace(marker, helper + marker, 1)
old = '''            semantic_observations = [
                observation
                for module, receipt in zip(members, receipts, strict=False)
                if isinstance(receipt, dict)
                and (
                    observation := _semantic_execution_observation(
                        module,
                        receipt,
                        dependent_ids=downstream_ids(module.module_id),
                    )
                )
                is not None
            ]'''
new = '''            semantic_observations = _semantic_execution_observations(
                members,
                receipts,
                downstream_ids=downstream_ids,
            )'''
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit("semantic observation replacement marker not found")
path.write_text(text, encoding="utf-8")

replace_once(
    "minecraft_mod_ai/extended_content_generator.py",
    '        "modules": [item["module_id"] for item in generation_records],\n        "catalog_module_count": committed_count,',
    '        "modules": [item["module_id"] for item in generation_records],\n        "module_ids": [module.module_id for module in selected],\n        "catalog_module_count": committed_count,',
)
replace_once(
    "minecraft_mod_ai/system_pack_generator.py",
    '        "pack_id": pack_id,\n        "input_definition_count": len(modules),',
    '        "pack_id": pack_id,\n        "module_ids": [str(item["module_id"]) for item in modules],\n        "input_definition_count": len(modules),',
)

Path("tests/test_semantic_receipt_ownership.py").write_text('''from minecraft_mod_ai.complete_orchestrator import _semantic_execution_observations
from minecraft_mod_ai.complete_spec import ProductionModule


def _module(module_id: str) -> ProductionModule:
    return ProductionModule(module_id=module_id, kind="item", config={"evidence_task": {"task_sha256": "sha256:" + module_id[0] * 64, "requirement_refs": [module_id], "gap_refs": [], "impact_probes": []}})


def _by_task(observations):
    return {item["task_id"]: (item["patch_receipt"], tuple(item["touched_paths"])) for item in observations}


def test_grouped_receipt_uses_explicit_module_ownership_not_position():
    members = [_module("alpha"), _module("beta"), _module("gamma")]
    grouped = {"module_ids": ["beta", "alpha"], "operation_count": 2, "touched_paths": ["src/grouped.java"], "patch_receipt": "grouped"}
    solo = {"module_id": "gamma", "operation_count": 1, "touched_paths": ["src/gamma.java"], "patch_receipt": "solo"}
    expected = {"alpha": ("grouped", ("src/grouped.java",)), "beta": ("grouped", ("src/grouped.java",)), "gamma": ("solo", ("src/gamma.java",))}
    forward = _semantic_execution_observations(members, [grouped, solo], downstream_ids=lambda _module_id: ())
    reverse = _semantic_execution_observations(members, [solo, grouped], downstream_ids=lambda _module_id: ())
    assert _by_task(forward) == expected
    assert _by_task(reverse) == expected


def test_unowned_receipt_is_not_positionally_assigned_in_multi_module_node():
    members = [_module("alpha"), _module("beta")]
    observations = _semantic_execution_observations(members, [{"patch_receipt": "ambiguous", "touched_paths": ["src/unknown.java"]}], downstream_ids=lambda _module_id: ())
    assert observations == []
''', encoding="utf-8")
