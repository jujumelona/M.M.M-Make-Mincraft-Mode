from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .scale_policy import ScalePolicy
from .world_compiler import (
    WorldCompileError,
    _palette,
    _sha256,
    _structure_nbt,
    compile_world_ir,
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
    larger than that artifact limit are partitioned into coherent templates and receive
    a sharded ``build_<id>`` function that places every piece at its approved offset.
    Small structures keep normal Jigsaw world-generation resources.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if not isinstance(ir, dict) or ir.get("schema_version") != "mmm/world-ir-v1":
        raise ScalableWorldCompileError("World IR schema is invalid.")

    expanded = dict(ir)
    expanded_structures: list[dict[str, Any]] = []
    logical: list[dict[str, Any]] = []

    for structure in ir.get("structures", []):
        if not isinstance(structure, dict):
            raise ScalableWorldCompileError("Every structure must be an object.")
        raw_size = structure.get("size", [9, 6, 9])
        size = _logical_size(raw_size)
        pieces = _partition(size, policy)
        if len(pieces) == 1:
            expanded_structures.append(dict(structure, size=list(size)))
            logical.append(
                {
                    "id": structure["id"],
                    "size": list(size),
                    "placement": "worldgen",
                    "pieces": [
                        {
                            "id": structure["id"],
                            "origin": [0, 0, 0],
                            "size": list(size),
                        }
                    ],
                }
            )
            continue

        piece_records: list[dict[str, Any]] = []
        for index, (origin, piece_size) in enumerate(pieces):
            piece_id = _piece_id(str(structure["id"]), index, origin)
            piece = dict(structure)
            piece["id"] = piece_id
            piece["size"] = list(piece_size)
            piece["jigsaw_depth"] = 1
            expanded_structures.append(piece)
            piece_records.append(
                {"id": piece_id, "origin": list(origin), "size": list(piece_size)}
            )
        logical.append(
            {
                "id": structure["id"],
                "size": list(size),
                "placement": "function_shards",
                "pieces": piece_records,
            }
        )

    expanded["structures"] = expanded_structures
    result = compile_world_ir(
        expanded,
        mod_id=mod_id,
        output_root=output_root,
        package_world_zip=False,
    )
    root = Path(result["output_root"]).resolve()

    original_by_id = {
        str(item["id"]): item
        for item in ir.get("structures", [])
        if isinstance(item, dict) and "id" in item
    }
    for logical_item in logical:
        if logical_item["placement"] != "function_shards":
            continue
        source = original_by_id[logical_item["id"]]
        palette = _palette(source)
        full_size = tuple(logical_item["size"])
        for piece in logical_item["pieces"]:
            origin = tuple(piece["origin"])
            piece_size = tuple(piece["size"])
            blocks = _logical_piece_blocks(source, full_size, origin, piece_size, palette)
            nbt_path = root / "data" / mod_id / "structures" / f"{piece['id']}.nbt"
            nbt_path.write_bytes(_structure_nbt(piece_size, palette, blocks))
        _remove_piece_worldgen(root, mod_id, logical_item["pieces"])
        _write_build_functions(root, mod_id, logical_item, policy)

    manifest_path = root / "mmm-world-manifest.json"
    manifest = {
        "schema_version": "mmm/world-compile-manifest-v2",
        "minecraft_version": "1.20.1",
        "mod_id": mod_id,
        "world_ir_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(ir, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "logical_structures": logical,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not path.is_symlink()
            and path != manifest_path
            and path.suffix != ".zip"
        ],
        "runtime_verification": "required",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    archive_path: Path | None = None
    if package_world_zip:
        import zipfile

        archive_path = root.with_suffix(".zip")
        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(root))

    return {
        "schema_version": "mmm/world-compile-result-v2",
        "output_root": str(root),
        "world_zip": str(archive_path) if archive_path else None,
        "structures": [item["id"] for item in logical],
        "logical_structure_count": len(logical),
        "physical_template_count": sum(len(item["pieces"]) for item in logical),
        "partitioned_structures": [
            item["id"] for item in logical if item["placement"] == "function_shards"
        ],
        "file_count": sum(1 for path in root.rglob("*") if path.is_file()),
        "manifest_sha256": _sha256(manifest_path),
        "runtime_verification": "required",
    }


def _logical_size(raw: Any) -> tuple[int, int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(type(value) is not int or value < 1 for value in raw)
    ):
        raise ScalableWorldCompileError(f"Invalid logical structure size: {raw!r}")
    return int(raw[0]), int(raw[1]), int(raw[2])


def _partition(
    size: tuple[int, int, int], policy: ScalePolicy
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    axis = policy.nbt_piece_axis
    pieces: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
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
                pieces.append(((x, y, z), piece))
    return pieces


def _piece_id(base: str, index: int, origin: tuple[int, int, int]) -> str:
    suffix = f"__p{index:05d}_{origin[0]}_{origin[1]}_{origin[2]}"
    candidate = base + suffix
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:12]
    return base[: 63 - len(digest) - 3] + "__" + digest


def _remove_piece_worldgen(root: Path, mod_id: str, pieces: list[dict[str, Any]]) -> None:
    for piece in pieces:
        piece_id = piece["id"]
        for relative in (
            f"data/{mod_id}/worldgen/template_pool/{piece_id}.json",
            f"data/{mod_id}/worldgen/processor_list/{piece_id}.json",
            f"data/{mod_id}/worldgen/structure/{piece_id}.json",
            f"data/{mod_id}/worldgen/structure_set/{piece_id}.json",
            f"data/{mod_id}/tags/worldgen/biome/has_structure/{piece_id}.json",
        ):
            (root / relative).unlink(missing_ok=True)


def _write_build_functions(
    root: Path,
    mod_id: str,
    logical: dict[str, Any],
    policy: ScalePolicy,
) -> None:
    function_root = root / "data" / mod_id / "functions"
    generated = function_root / "generated" / logical["id"]
    generated.mkdir(parents=True, exist_ok=True)
    commands = [
        f"place template {mod_id}:{piece['id']} ~{piece['origin'][0]} ~{piece['origin'][1]} ~{piece['origin'][2]}"
        for piece in logical["pieces"]
    ]
    shard_paths: list[str] = []
    for index in range(0, len(commands), policy.function_shard_size):
        shard_index = index // policy.function_shard_size
        path = generated / f"part_{shard_index:04d}.mcfunction"
        path.write_text(
            "\n".join(commands[index : index + policy.function_shard_size]) + "\n",
            encoding="utf-8",
        )
        shard_paths.append(f"{mod_id}:generated/{logical['id']}/part_{shard_index:04d}")
    root_function = function_root / f"build_{logical['id']}.mcfunction"
    root_function.write_text(
        "\n".join(f"function {path}" for path in shard_paths) + "\n",
        encoding="utf-8",
    )


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
