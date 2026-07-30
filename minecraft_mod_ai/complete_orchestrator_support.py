from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any

from .complete_spec import CompleteProposal, ProductionModule


class CompleteProductionError(RuntimeError):
    pass


def _locate_existing_fabric_root(extracted_root: Path) -> Path:
    direct = extracted_root / "src/main/resources/fabric.mod.json"
    if direct.is_file() and not direct.is_symlink():
        return extracted_root
    candidates = sorted(
        path.parent.parent.parent.parent
        for path in extracted_root.rglob("fabric.mod.json")
        if path.as_posix().endswith("src/main/resources/fabric.mod.json")
        and path.is_file()
        and not path.is_symlink()
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(extracted_root)
        except ValueError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if len(unique) != 1:
        raise CompleteProductionError(
            "Existing source ZIP must contain exactly one Fabric project root; "
            f"found {len(unique)}."
        )
    return unique[0]


def _topological_modules(
    modules: tuple[ProductionModule, ...] | list[ProductionModule],
) -> list[ProductionModule]:
    lookup = {module.module_id: module for module in modules}
    indegree = {module.module_id: len(module.depends_on) for module in modules}
    outgoing: dict[str, list[str]] = {module.module_id: [] for module in modules}
    for module in modules:
        for dependency in module.depends_on:
            if dependency not in lookup:
                raise CompleteProductionError(
                    f"Production module {module.module_id} references missing {dependency}."
                )
            outgoing[dependency].append(module.module_id)
    ready = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[ProductionModule] = []
    while ready:
        node = heapq.heappop(ready)
        ordered.append(lookup[node])
        for dependent in outgoing[node]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(lookup):
        raise CompleteProductionError(
            "Production module graph contains an unresolved cycle."
        )
    return ordered


def _normalize_modules(
    modules: tuple[ProductionModule, ...], spec
) -> tuple[list[ProductionModule], list[dict[str, Any]]]:
    base = {content.content_id: content.kind.value for content in spec.contents}
    if spec.boss is not None:
        base[spec.boss.entity_id] = "boss"
        base[f"{spec.boss.entity_id}_spawn_egg"] = "item"
    if spec.arena is not None:
        base[spec.arena.arena_id] = "structure"
    reused: set[str] = set()
    staged: list[ProductionModule] = []
    receipts: list[dict[str, Any]] = []
    for module in modules:
        existing = base.get(module.module_id)
        if existing is None:
            staged.append(module)
        elif existing == module.kind or {existing, module.kind} <= {"entity", "boss"}:
            reused.add(module.module_id)
            receipts.append(
                {
                    "schema_version": "mmm/bootstrap-dedup-v1",
                    "status": "REUSED",
                    "module_id": module.module_id,
                    "kind": module.kind,
                }
            )
        else:
            raise CompleteProductionError(
                f"Module {module.module_id}/{module.kind} collides with bootstrap {existing}."
            )
    kept = [
        ProductionModule(
            module_id=module.module_id,
            kind=module.kind,
            config=module.config,
            depends_on=tuple(dep for dep in module.depends_on if dep not in reused),
            required_gates=module.required_gates,
        )
        for module in staged
    ]
    return _topological_modules(kept), receipts


def _system_groups(
    modules: list[ProductionModule],
) -> dict[str, list[ProductionModule]]:
    mapping = {
        "quest": "quest-system",
        "class": "class-skill-system",
        "skill": "class-skill-system",
        "economy": "economy-shop",
        "shop": "economy-shop",
        "gui": "gui-networking",
        "networking": "gui-networking",
        "party": "party-guild",
        "guild": "party-guild",
    }
    result: dict[str, list[ProductionModule]] = {}
    for module in modules:
        pack = mapping.get(module.kind)
        if pack:
            result.setdefault(pack, []).append(module)
    return result


def _handled_module_ids(modules: list[ProductionModule]) -> set[str]:
    built_in = {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "machine",
        "effect",
        "enchantment",
        "command",
        "recipe",
        "advancement",
        "loot",
        "quest",
        "class",
        "skill",
        "economy",
        "shop",
        "gui",
        "networking",
        "party",
        "guild",
        "entity",
        "boss",
        "npc",
        "structure",
        "audio",
    }
    return {module.module_id for module in modules if module.kind in built_in}


def _module_dict(module: ProductionModule) -> dict[str, Any]:
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "config": module.config,
        "depends_on": list(module.depends_on),
        "required_gates": list(module.required_gates),
    }


def _jar_path(build: dict[str, Any]) -> Path:
    value = build.get("jar_path")
    if not isinstance(value, str):
        raise CompleteProductionError("Gradle report did not contain a JAR path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CompleteProductionError("Gradle JAR path is missing or unsafe.")
    return path


def _external_gates(proposal: CompleteProposal, options: Any) -> list[str]:
    gates = ["Gradle", "GameTest", "JAR validation"]
    if proposal.external_runtime_required:
        gates.extend(
            [
                "Minecraft server/client runtime",
                "Mineflayer playtest",
                "visual review",
            ]
        )
    if any(
        module.kind in {"entity", "boss", "npc"}
        for module in proposal.modules
    ):
        gates.append("Blockbench UV/render review")
    return gates


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CompleteProductionError("Model response did not contain a JSON object.")
