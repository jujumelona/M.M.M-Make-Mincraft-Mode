from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path
from typing import Any

from .complete_spec import CompleteProposal, ProductionModule
from .platform_backend_contract import (
    ENTITY_PIPELINE_KINDS,
    EXTENDED_CONTENT_KINDS,
    SYSTEM_KIND_TO_PACK,
)


def file_sha256(path: Path) -> str:
    """Hash one artifact without loading the complete file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class CompleteProductionError(RuntimeError):
    """Production failure with optional explicit validation checkpoint provenance.

    A display message must not be used as a control-plane retry identifier.
    """

    def __init__(self, message: str, *, checkpoint_id: str | None = None) -> None:
        super().__init__(message)
        self.checkpoint_id = checkpoint_id

def _locate_existing_fabric_root(extracted_root: Path) -> Path:
    direct = extracted_root / 'src/main/resources/fabric.mod.json'
    if direct.is_file() and (not direct.is_symlink()):
        return extracted_root
    candidates = sorted(path.parent.parent.parent.parent for path in extracted_root.rglob('fabric.mod.json') if path.as_posix().endswith('src/main/resources/fabric.mod.json') and path.is_file() and (not path.is_symlink()))
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
        raise CompleteProductionError(f'Existing source ZIP must contain exactly one Fabric project root; found {len(unique)}.')
    return unique[0]

def _topological_modules(modules: tuple[ProductionModule, ...] | list[ProductionModule]) -> list[ProductionModule]:
    lookup = {module.module_id: module for module in modules}
    if len(lookup) != len(modules):
        raise CompleteProductionError('Production module IDs must be unique.')
    indegree = {module.module_id: len(module.depends_on) for module in modules}
    outgoing: dict[str, list[str]] = {module.module_id: [] for module in modules}
    for module in modules:
        for dependency in module.depends_on:
            if dependency not in lookup:
                raise CompleteProductionError(f'Production module {module.module_id} references missing {dependency}.')
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
        raise CompleteProductionError('Production module graph contains an unresolved cycle.')
    return ordered

def _is_custom(module: ProductionModule) -> bool:
    return module.kind == 'custom_java' or module.config.get('implementation') == 'custom'

def _normalize_modules(
    modules: tuple[ProductionModule, ...],
    spec,
) -> tuple[list[ProductionModule], list[dict[str, Any]]]:
    """Return the approved production graph unchanged after defensive validation.

    Bootstrap/content ownership collisions are invalid at the CompleteProposal boundary.
    Execution must never rename, deduplicate, or remove dependencies from an approved
    module graph.
    """

    bootstrap = {content.content_id for content in spec.contents}
    if spec.boss is not None:
        bootstrap.update(
            {
                spec.boss.entity_id,
                f"{spec.boss.entity_id}_spawn_egg",
            }
        )

    collisions = sorted(
        bootstrap & {module.module_id for module in modules}
    )
    if collisions:
        raise CompleteProductionError(
            "PRODUCTION_BOOTSTRAP_OWNERSHIP_COLLISION: "
            + ", ".join(collisions[:20])
        )

    for module in modules:
        if _is_custom(module):
            raise CompleteProductionError(
                "CUSTOM_JAVA_BACKEND_REMOVED: "
                f"{module.module_id} must be expressed as typed_host or a supported "
                "deterministic production module; model-backed custom Java generation "
                "is not a production route."
            )

    return (_topological_modules(list(modules)), [])



def _system_groups(modules: list[ProductionModule]) -> dict[str, list[ProductionModule]]:
    result: dict[str, list[ProductionModule]] = {}
    for module in modules:
        pack = SYSTEM_KIND_TO_PACK.get(module.kind)
        if pack:
            result.setdefault(pack, []).append(module)
    return result

def _handled_module_ids(modules: list[ProductionModule]) -> set[str]:
    built_in = (
        set(EXTENDED_CONTENT_KINDS)
        | set(SYSTEM_KIND_TO_PACK)
        | set(ENTITY_PIPELINE_KINDS)
    )
    return {module.module_id for module in modules if module.kind in built_in}

def _module_dict(module: ProductionModule) -> dict[str, Any]:
    return {'module_id': module.module_id, 'kind': module.kind, 'config': module.config, 'depends_on': list(module.depends_on), 'required_gates': list(module.required_gates)}

def _jar_path(build: dict[str, Any]) -> Path:
    value = build.get('jar_path')
    if not isinstance(value, str):
        raise CompleteProductionError('Gradle report did not contain a JAR path.')
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CompleteProductionError('Gradle JAR path is missing or unsafe.')
    return path

def _external_gates(proposal: CompleteProposal, options: Any) -> list[str]:
    gates = ['Gradle', 'GameTest', 'JAR validation']
    if proposal.external_runtime_required:
        gates.extend(['Minecraft server/client runtime', 'Mineflayer playtest', 'visual review'])
    if any(module.kind in {'entity', 'boss', 'npc'} for module in proposal.modules):
        gates.append('Blockbench UV/render review')
    return gates

def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise CompleteProductionError('Model response did not contain a JSON object.')
