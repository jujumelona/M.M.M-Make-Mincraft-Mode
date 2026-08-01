from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .scale_policy import ScalePolicy
from .world_compiler import (
    WorldCompileError,
    _build_blocks,
    _compiled_root_matches,
    _package_world_root,
    _palette,
    _placement_fields,
    _prepare_world_stage,
    _promote_world_stage,
    _structure_nbt,
    _validate,
    _world_ir_sha256,
    _world_result,
    _write_pack_metadata,
    _write_structure_resources,
    _write_structure_runtime_contract,
    _write_world_contract_shards,
    _write_world_manifest,
    _write_json,
)


class ScalableWorldCompileError(WorldCompileError):
    pass


def compile_scalable_world_ir(
    ir: dict[str, Any],
    *,
    mod_id: str,
    output_root: str | Path,
    package_world_zip: bool = True,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    """Compile World IR without rejecting large logical structures.

    Vanilla structure templates remain bounded physical artifacts. Logical structures
    larger than that artifact limit are streamed into coherent templates and bounded
    function pages. The generated Fabric runtime executes those pages at one persistent
    anchor across ticks. Small structures keep normal Jigsaw world-generation resources.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if not isinstance(ir, dict) or ir.get("schema_version") != "mmm/world-ir-v1":
        raise ScalableWorldCompileError("World IR schema is invalid.")

    structures = sorted(
        ir.get("structures", []),
        key=lambda item: str(item.get("id", ""))
        if isinstance(item, dict)
        else "",
    )
    validation_ir = {
        **ir,
        "structures": [
            {**item, "size": [1, 1, 1]}
            if isinstance(item, dict)
            else item
            for item in structures
        ],
    }
    _validate(validation_ir, mod_id)
    world_hash = _world_ir_sha256(ir)
    root = Path(output_root).expanduser().resolve()
    logical_ids = [str(item["id"]) for item in structures]
    if _compiled_root_matches(
        root,
        mod_id=mod_id,
        world_ir_sha256=world_hash,
    ):
        archive = _package_world_root(root) if package_world_zip else None
        result = _world_result(root, archive, structures=logical_ids)
        manifest = json.loads(
            (root / "mmm-world-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        result.update(
            {
                "logical_structure_count": len(logical_ids),
                "physical_template_count": int(
                    manifest.get("physical_template_count", len(logical_ids))
                ),
                "partitioned_structures": _partitioned_contract_ids(
                    root,
                    mod_id,
                ),
                "runtime_bridge_required": bool(
                    manifest.get("partitioned_structure_count", 0)
                ),
            }
        )
        return result

    stage = _prepare_world_stage(
        root,
        mod_id=mod_id,
        world_ir_sha256=world_hash,
    )
    if stage.exists():
        _promote_world_stage(stage, root)
        archive = _package_world_root(root) if package_world_zip else None
        result = _world_result(
            root,
            archive,
            structures=logical_ids,
        )
        manifest = json.loads(
            (root / "mmm-world-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        result.update(
            {
                "logical_structure_count": len(logical_ids),
                "physical_template_count": int(
                    manifest.get(
                        "physical_template_count",
                        len(logical_ids),
                    )
                ),
                "partitioned_structures": _partitioned_contract_ids(
                    root,
                    mod_id,
                ),
                "runtime_bridge_required": bool(
                    manifest.get("partitioned_structure_count", 0)
                ),
            }
        )
        return result
    stage.mkdir(parents=True, exist_ok=False)
    data_root = stage / "data" / mod_id
    partitioned: list[str] = []
    physical_template_count = 0
    for structure_index, structure in enumerate(structures):
        if not isinstance(structure, dict):
            raise ScalableWorldCompileError(
                "Every structure must be an object."
            )
        size = _logical_size(structure.get("size", [9, 6, 9]))
        piece_count = _partition_count(size, policy)
        palette = _palette(structure)
        if piece_count == 1:
            _write_structure_resources(
                data_root=data_root,
                mod_id=mod_id,
                structure=structure,
                structure_index=structure_index,
                size=size,
                palette=palette,
                blocks=_build_blocks(structure, size, palette),
                emit_worldgen=True,
                runtime_placement="vanilla_worldgen",
            )
            physical_template_count += 1
            continue

        structure_id = str(structure["id"])
        partitioned.append(structure_id)
        shard_count, written_pieces = _write_partitioned_structure(
            root=stage,
            data_root=data_root,
            mod_id=mod_id,
            structure=structure,
            structure_index=structure_index,
            full_size=size,
            palette=palette,
            policy=policy,
        )
        physical_template_count += written_pieces
        spacing, separation, salt, biomes = _placement_fields(
            structure,
            structure_index,
        )
        _write_structure_runtime_contract(
            data_root=data_root,
            mod_id=mod_id,
            structure=structure,
            placement="runtime_function_shards",
            spacing=spacing,
            separation=separation,
            salt=salt,
            biomes=biomes,
            shard_count=shard_count,
        )

    _write_world_contract_shards(ir, data_root=data_root)
    _write_pack_metadata(stage)
    _write_world_manifest(
        stage,
        mod_id=mod_id,
        world_ir_sha256=world_hash,
        logical_structure_count=len(logical_ids),
        partitioned_structure_count=len(partitioned),
        physical_template_count=physical_template_count,
    )
    if not _compiled_root_matches(
        stage,
        mod_id=mod_id,
        world_ir_sha256=world_hash,
    ):
        raise ScalableWorldCompileError(
            "Staged scalable world output failed integrity validation."
        )
    _promote_world_stage(stage, root)
    archive = _package_world_root(root) if package_world_zip else None
    result = _world_result(root, archive, structures=logical_ids)
    result.update(
        {
            "logical_structure_count": len(logical_ids),
            "physical_template_count": physical_template_count,
            "partitioned_structures": partitioned,
            "runtime_bridge_required": bool(partitioned),
        }
    )
    return result


def _logical_size(raw: Any) -> tuple[int, int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(type(value) is not int or value < 1 for value in raw)
    ):
        raise ScalableWorldCompileError(f"Invalid logical structure size: {raw!r}")
    return int(raw[0]), int(raw[1]), int(raw[2])


def _partition_count(
    size: tuple[int, int, int],
    policy: ScalePolicy,
) -> int:
    axis = policy.nbt_piece_axis
    return math.prod(
        (value + axis - 1) // axis
        for value in size
    )


def _partition(
    size: tuple[int, int, int],
    policy: ScalePolicy,
):
    axis = policy.nbt_piece_axis
    for x in range(0, size[0], axis):
        for y in range(0, size[1], axis):
            for z in range(0, size[2], axis):
                piece = (
                    min(axis, size[0] - x),
                    min(axis, size[1] - y),
                    min(axis, size[2] - z),
                )
                if piece[0] * piece[1] * piece[2] > policy.nbt_piece_volume:
                    raise ScalableWorldCompileError(
                        "Configured NBT piece exceeds the physical piece-volume policy."
                    )
                yield (x, y, z), piece


def _piece_id(base: str, index: int, origin: tuple[int, int, int]) -> str:
    suffix = f"__p{index:05d}_{origin[0]}_{origin[1]}_{origin[2]}"
    candidate = base + suffix
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:12]
    return base[: 63 - len(digest) - 3] + "__" + digest


def _write_partitioned_structure(
    *,
    root: Path,
    data_root: Path,
    mod_id: str,
    structure: dict[str, Any],
    structure_index: int,
    full_size: tuple[int, int, int],
    palette: list[str],
    policy: ScalePolicy,
) -> tuple[int, int]:
    del structure_index
    structure_id = str(structure["id"])
    generated = (
        root
        / "data"
        / mod_id
        / "functions"
        / "generated"
        / structure_id
    )
    generated.mkdir(parents=True, exist_ok=True)
    page: list[str] = []
    per_tick = min(
        policy.function_shard_size,
        policy.world_placements_per_tick,
    )
    shard_count = 0
    piece_count = 0
    for piece_index, (origin, piece_size) in enumerate(
        _partition(full_size, policy)
    ):
        piece_id = _piece_id(structure_id, piece_index, origin)
        blocks = _logical_piece_blocks(
            structure,
            full_size,
            origin,
            piece_size,
            palette,
        )
        nbt_path = (
            data_root / "structures" / f"{piece_id}.nbt"
        )
        nbt_path.parent.mkdir(parents=True, exist_ok=True)
        nbt_path.write_bytes(
            _structure_nbt(piece_size, palette, blocks)
        )
        _write_json(
            data_root
            / "mmm_world"
            / "contracts"
            / "structure_pieces"
            / structure_id
            / f"{piece_id}.json",
            {
                "schema_version": "mmm/world-structure-piece-v1",
                "logical_structure_id": structure_id,
                "template": f"{mod_id}:{piece_id}",
                "origin": list(origin),
                "size": list(piece_size),
            },
        )
        page.append(
            f"place template {mod_id}:{piece_id} "
            f"~{origin[0]} ~{origin[1]} ~{origin[2]}"
        )
        piece_count += 1
        if len(page) == per_tick:
            _write_function_page(generated, shard_count, page)
            shard_count += 1
            page = []
    if page:
        _write_function_page(generated, shard_count, page)
        shard_count += 1
    if not shard_count or not piece_count:
        raise ScalableWorldCompileError(
            f"Partitioned structure emitted no pieces: {structure_id}"
        )
    return shard_count, piece_count


def _write_function_page(
    generated: Path,
    shard_index: int,
    commands: list[str],
) -> None:
    (generated / f"part_{shard_index:04d}.mcfunction").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )


def _partitioned_contract_ids(root: Path, mod_id: str) -> list[str]:
    contract_root = (
        root
        / "data"
        / mod_id
        / "mmm_world"
        / "contracts"
        / "structures"
    )
    result: list[str] = []
    for path in sorted(contract_root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if value.get("placement") == "runtime_function_shards":
            result.append(str(value.get("id", path.stem)))
    return result


def _logical_piece_blocks(
    structure: dict[str, Any],
    full_size: tuple[int, int, int],
    origin: tuple[int, int, int],
    piece_size: tuple[int, int, int],
    palette: list[str],
) -> list[tuple[tuple[int, int, int], int]]:
    air = palette.index("minecraft:air") if "minecraft:air" in palette else None
    solids = [index for index, value in enumerate(palette) if value != "minecraft:air"]
    primary = solids[0] if solids else 0
    accent = solids[1] if len(solids) > 1 else primary
    kind = str(structure.get("kind", "room")).lower()
    width, height, depth = full_size
    result: list[tuple[tuple[int, int, int], int]] = []

    for lx in range(piece_size[0]):
        for ly in range(piece_size[1]):
            for lz in range(piece_size[2]):
                x, y, z = origin[0] + lx, origin[1] + ly, origin[2] + lz
                state = _logical_state(kind, x, y, z, width, height, depth, primary, accent, air)
                if state is not None:
                    result.append(((lx, ly, lz), state))
    return result


def _logical_state(
    kind: str,
    x: int,
    y: int,
    z: int,
    width: int,
    height: int,
    depth: int,
    primary: int,
    accent: int,
    air: int | None,
) -> int | None:
    if y == 0:
        if kind in {"road", "bridge"}:
            center = width // 2
            half = max(1, min(width // 2, 2))
            return primary if abs(x - center) <= half else air
        return primary

    if kind in {"road", "bridge"}:
        center = width // 2
        if y == 1 and z % 4 == 0 and abs(x - center) == 3:
            return accent
        return air

    if kind == "tower":
        cx, cz = (width - 1) / 2.0, (depth - 1) / 2.0
        radius = max(2.0, min(width, depth) / 2.0 - 1.0)
        distance = math.hypot(x - cx, z - cz)
        wall = radius - 0.8 <= distance <= radius + 0.8
        roof = y == height - 1 and distance <= radius
        stair = abs((math.atan2(z - cz, x - cx) + math.pi) * radius - y * 1.7) < 0.8
        if roof or stair:
            return accent
        if wall:
            doorway = z == 0 and abs(x - int(cx)) <= 1 and y <= 2
            window = y % 4 == 2 and (x + z) % 5 == 0
            return air if doorway or window else primary
        return air

    boundary = x in {0, width - 1} or z in {0, depth - 1}
    roof = y == height - 1
    doorway = z == 0 and abs(x - width // 2) <= 1 and y <= 2
    window = boundary and y in {2, 3} and (x + z) % 5 == 0
    partition = (
        kind in {"village", "dungeon", "room", "arena"}
        and y < height - 1
        and (x % 12 == 0 or z % 12 == 0)
        and not (abs(x - width // 2) <= 1 or abs(z - depth // 2) <= 1)
    )
    landmark = kind in {"village", "arena"} and x == width // 2 and z == depth // 2 and y < min(height, 8)
    if doorway or window:
        return air
    if roof:
        return accent if (x + z) % 7 == 0 else primary
    if boundary or partition:
        return primary
    if landmark:
        return accent
    return air
