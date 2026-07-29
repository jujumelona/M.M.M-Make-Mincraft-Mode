from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any, Iterable


_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_BLOCK = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_DATA_VERSION_1201 = 3465


class WorldCompileError(ValueError):
    pass


def compile_world_ir(
    ir: dict[str, Any],
    *,
    mod_id: str,
    output_root: str | Path,
    package_world_zip: bool = True,
) -> dict[str, Any]:
    """Compile validated WorldDesignIR into deterministic 1.20.1 datapack resources.

    The compiler emits real binary structure NBT plus Jigsaw template pools,
    processors, structures, structure sets and biome tags. Runtime placement still
    has to pass a disposable-world integration test before release.
    """

    _validate(ir, mod_id)
    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    data_root = root / "data" / mod_id
    generated: list[Path] = []
    structures = sorted(ir["structures"], key=lambda item: item["id"])
    for index, structure in enumerate(structures):
        structure_id = structure["id"]
        size = _size(structure)
        palette = _palette(structure)
        blocks = _build_blocks(structure, size, palette)
        nbt_path = data_root / "structures" / f"{structure_id}.nbt"
        nbt_path.parent.mkdir(parents=True, exist_ok=True)
        nbt_path.write_bytes(_structure_nbt(size, palette, blocks))
        generated.append(nbt_path)

        pool = {
            "name": f"{mod_id}:{structure_id}",
            "fallback": "minecraft:empty",
            "elements": [
                {
                    "weight": 1,
                    "element": {
                        "location": f"{mod_id}:{structure_id}",
                        "processors": f"{mod_id}:{structure_id}",
                        "projection": "rigid",
                        "element_type": "minecraft:single_pool_element",
                    },
                }
            ],
        }
        generated.append(
            _write_json(
                data_root / "worldgen" / "template_pool" / f"{structure_id}.json",
                pool,
            )
        )
        generated.append(
            _write_json(
                data_root / "worldgen" / "processor_list" / f"{structure_id}.json",
                {"processors": []},
            )
        )
        structure_json = {
            "type": "minecraft:jigsaw",
            "biomes": f"#{mod_id}:has_structure/{structure_id}",
            "step": "surface_structures",
            "spawn_overrides": {},
            "terrain_adaptation": "beard_thin",
            "start_pool": f"{mod_id}:{structure_id}",
            "size": int(structure.get("jigsaw_depth", 1)),
            "start_height": {
                "absolute": int(structure.get("start_height", 0))
            },
            "project_start_to_heightmap": "WORLD_SURFACE_WG",
            "max_distance_from_center": int(
                structure.get("max_distance_from_center", 80)
            ),
            "use_expansion_hack": False,
        }
        generated.append(
            _write_json(
                data_root / "worldgen" / "structure" / f"{structure_id}.json",
                structure_json,
            )
        )
        spacing = int(structure.get("spacing", 32))
        separation = int(structure.get("separation", 8))
        if not 2 <= separation < spacing <= 4096:
            raise WorldCompileError(
                f"Invalid spacing/separation for {structure_id}: {spacing}/{separation}"
            )
        generated.append(
            _write_json(
                data_root / "worldgen" / "structure_set" / f"{structure_id}.json",
                {
                    "structures": [
                        {
                            "structure": f"{mod_id}:{structure_id}",
                            "weight": 1,
                        }
                    ],
                    "placement": {
                        "type": "minecraft:random_spread",
                        "salt": int(structure.get("salt", 918273 + index)),
                        "spacing": spacing,
                        "separation": separation,
                    },
                },
            )
        )
        biomes = structure.get(
            "biomes",
            ["minecraft:plains", "minecraft:forest"],
        )
        if not isinstance(biomes, list) or not biomes:
            raise WorldCompileError(f"{structure_id} requires at least one biome.")
        generated.append(
            _write_json(
                data_root
                / "tags"
                / "worldgen"
                / "biome"
                / "has_structure"
                / f"{structure_id}.json",
                {"replace": False, "values": biomes},
            )
        )

    generated.append(
        _write_json(
            root / "pack.mcmeta",
            {
                "pack": {
                    "pack_format": 15,
                    "description": (
                        "M.M.M generated Minecraft 1.20.1 worldgen resources; "
                        "runtime verification required"
                    ),
                }
            },
        )
    )
    manifest = {
        "schema_version": "mmm/world-compile-manifest-v1",
        "minecraft_version": "1.20.1",
        "mod_id": mod_id,
        "world_ir_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(ir, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "files": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for path in sorted(generated)
        ],
        "runtime_verification": "required",
    }
    manifest_path = _write_json(root / "mmm-world-manifest.json", manifest)
    generated.append(manifest_path)

    archive_path: Path | None = None
    if package_world_zip:
        archive_path = root.with_suffix(".zip")
        if archive_path.exists():
            raise FileExistsError(archive_path)
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    archive.write(path, path.relative_to(root))

    return {
        "schema_version": "mmm/world-compile-result-v1",
        "output_root": str(root),
        "world_zip": str(archive_path) if archive_path else None,
        "structures": [item["id"] for item in structures],
        "file_count": len(generated),
        "manifest_sha256": _sha256(manifest_path),
        "runtime_verification": "required",
    }


def _validate(ir: dict[str, Any], mod_id: str) -> None:
    if not _ID.fullmatch(mod_id):
        raise WorldCompileError("Invalid mod_id.")
    required = {
        "schema_version",
        "regions",
        "routes",
        "structures",
        "quests",
        "constraints",
    }
    if set(ir) != required or ir.get("schema_version") != "mmm/world-ir-v1":
        raise WorldCompileError("World IR schema is invalid.")
    for key in ("regions", "routes", "structures", "quests", "constraints"):
        if not isinstance(ir[key], list):
            raise WorldCompileError(f"World IR field {key} must be a list.")
    region_ids = set()
    for region in ir["regions"]:
        if not isinstance(region, dict) or not _ID.fullmatch(str(region.get("id", ""))):
            raise WorldCompileError("Invalid region.")
        if region["id"] in region_ids:
            raise WorldCompileError(f"Duplicate region: {region['id']}")
        region_ids.add(region["id"])
    graph: dict[str, set[str]] = {region: set() for region in region_ids}
    for route in ir["routes"]:
        if not isinstance(route, dict):
            raise WorldCompileError("Invalid route.")
        left, right = route.get("from"), route.get("to")
        if left not in region_ids or right not in region_ids:
            raise WorldCompileError("Route references unknown region.")
        graph[left].add(right)
        graph[right].add(left)
    if region_ids and not _connected(graph):
        raise WorldCompileError("World region graph is disconnected.")
    structure_ids = set()
    for structure in ir["structures"]:
        if not isinstance(structure, dict):
            raise WorldCompileError("Invalid structure.")
        structure_id = str(structure.get("id", ""))
        if not _ID.fullmatch(structure_id) or structure_id in structure_ids:
            raise WorldCompileError(f"Invalid or duplicate structure id: {structure_id}")
        structure_ids.add(structure_id)
        if structure.get("region_id") not in region_ids:
            raise WorldCompileError(
                f"Structure {structure_id} references an unknown region."
            )
        for block in _palette(structure):
            if not _BLOCK.fullmatch(block):
                raise WorldCompileError(f"Invalid palette block: {block}")


def _connected(graph: dict[str, set[str]]) -> bool:
    if not graph:
        return True
    seen: set[str] = set()
    pending = [next(iter(graph))]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(graph[node] - seen)
    return seen == set(graph)


def _size(structure: dict[str, Any]) -> tuple[int, int, int]:
    raw = structure.get("size", [9, 6, 9])
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or any(type(value) is not int or not 1 <= value <= 48 for value in raw)
    ):
        raise WorldCompileError(f"Invalid structure size: {raw!r}")
    volume = raw[0] * raw[1] * raw[2]
    if volume > 32768:
        raise WorldCompileError("Structure volume exceeds 32,768 blocks.")
    return raw[0], raw[1], raw[2]


def _palette(structure: dict[str, Any]) -> list[str]:
    raw = structure.get(
        "palette",
        ["minecraft:stone_bricks", "minecraft:air"],
    )
    if not isinstance(raw, list) or not 1 <= len(raw) <= 64:
        raise WorldCompileError("Structure palette must contain 1-64 block IDs.")
    return [str(value) for value in raw]


def _build_blocks(
    structure: dict[str, Any],
    size: tuple[int, int, int],
    palette: list[str],
) -> list[tuple[tuple[int, int, int], int]]:
    """Build a deterministic architecture matching the requested structure kind.

    The previous compiler emitted only a hollow cuboid. This compiler creates floors,
    entrances, windows, roofs, internal rooms/corridors, stairs and landmarks while
    remaining bounded to the vanilla structure NBT volume.
    """

    width, height, depth = size
    air_state = palette.index("minecraft:air") if "minecraft:air" in palette else None
    solid_states = [index for index, block in enumerate(palette) if block != "minecraft:air"]
    primary = solid_states[0] if solid_states else 0
    accent = solid_states[1] if len(solid_states) > 1 else primary
    blocks: dict[tuple[int, int, int], int] = {}
    if air_state is not None:
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    blocks[(x, y, z)] = air_state

    def put(x: int, y: int, z: int, state: int = primary) -> None:
        if 0 <= x < width and 0 <= y < height and 0 <= z < depth:
            blocks[(x, y, z)] = state

    def floor(y: int = 0, state: int = primary) -> None:
        for x in range(width):
            for z in range(depth):
                put(x, y, z, state)

    kind = str(structure.get("kind", "room")).lower()
    floor()

    if kind in {"road", "bridge"}:
        center = width // 2
        half = max(1, min(width // 2, int(structure.get("road_half_width", 2))))
        for x in range(max(0, center - half), min(width, center + half + 1)):
            for z in range(depth):
                put(x, 0, z, primary if (x + z) % 3 else accent)
        for z in range(0, depth, 4):
            put(max(0, center - half - 1), 1, z, accent)
            put(min(width - 1, center + half + 1), 1, z, accent)
    elif kind == "tower":
        cx, cz = (width - 1) / 2.0, (depth - 1) / 2.0
        radius = max(2.0, min(width, depth) / 2.0 - 1.0)
        for y in range(1, height):
            for x in range(width):
                for z in range(depth):
                    distance = math.hypot(x - cx, z - cz)
                    if radius - 0.8 <= distance <= radius + 0.25:
                        if not (z == 0 and abs(x - cx) <= 1 and y <= 2):
                            put(x, y, z, primary)
            if y % 4 == 0:
                for x in range(width):
                    for z in range(depth):
                        if math.hypot(x - cx, z - cz) < radius - 0.8:
                            put(x, y, z, accent)
        for step in range(min(height - 1, max(width, depth))):
            put(1 + step % max(1, width - 2), 1 + step, 1 + (step // max(1, width - 2)) % max(1, depth - 2), accent)
    elif kind == "dungeon":
        for x in range(width):
            for y in range(1, height):
                for z in range(depth):
                    boundary = x in {0, width - 1} or z in {0, depth - 1} or y == height - 1
                    if boundary and not (z == 0 and abs(x - width // 2) <= 1 and y <= 2):
                        put(x, y, z, primary)
        mid_x, mid_z = width // 2, depth // 2
        for x in range(1, width - 1):
            if abs(x - mid_x) > 1:
                for y in range(1, min(height - 1, 4)):
                    put(x, y, mid_z, accent)
        for z in range(1, depth - 1):
            if abs(z - mid_z) > 1:
                for y in range(1, min(height - 1, 4)):
                    put(mid_x, y, z, accent)
        for x, z in ((1, 1), (width - 2, 1), (1, depth - 2), (width - 2, depth - 2)):
            for y in range(1, height - 1):
                put(x, y, z, accent)
    elif kind in {"arena", "boss_arena"}:
        for x in range(width):
            for z in range(depth):
                if x in {0, width - 1} or z in {0, depth - 1}:
                    for y in range(1, min(height, 5)):
                        if not (z == 0 and abs(x - width // 2) <= 1 and y <= 2):
                            put(x, y, z, primary)
                elif (x + z) % 5 == 0:
                    put(x, 0, z, accent)
        for x, z in ((2, 2), (width - 3, 2), (2, depth - 3), (width - 3, depth - 3)):
            for y in range(1, min(height, 4)):
                put(x, y, z, accent)
    else:
        # Village/house/default: usable doorway, windows, pitched roof and an interior partition.
        wall_top = max(2, height - 2)
        for x in range(width):
            for z in range(depth):
                boundary = x in {0, width - 1} or z in {0, depth - 1}
                if not boundary:
                    continue
                for y in range(1, wall_top + 1):
                    doorway = z == 0 and abs(x - width // 2) <= 1 and y <= 2
                    window = y == 2 and ((z in {0, depth - 1} and x % 4 == 1) or (x in {0, width - 1} and z % 4 == 1))
                    if not doorway and not window:
                        put(x, y, z, primary)
        partition_x = width // 2
        for z in range(2, depth - 1):
            if abs(z - depth // 2) > 1:
                for y in range(1, min(wall_top, 3) + 1):
                    put(partition_x, y, z, accent)
        roof_y = min(height - 1, wall_top + 1)
        for layer in range(max(1, min(width, depth) // 3)):
            y = min(height - 1, roof_y + layer)
            for x in range(layer, width - layer):
                put(x, y, layer, accent)
                put(x, y, depth - 1 - layer, accent)
        for x, z in ((1, 1), (width - 2, depth - 2)):
            for y in range(1, min(height - 1, 3)):
                put(x, y, z, accent)

    return sorted(blocks.items())


def _structure_nbt(
    size: tuple[int, int, int],
    palette: list[str],
    blocks: list[tuple[tuple[int, int, int], int]],
) -> bytes:
    payload = bytearray()
    payload.extend(_named_tag(3, "DataVersion", _int_payload(_DATA_VERSION_1201)))
    payload.extend(
        _named_tag(
            9,
            "size",
            _list_payload(3, [_int_payload(value) for value in size]),
        )
    )
    palette_payloads = [
        _compound_payload(
            [_named_tag(8, "Name", _string_payload(block))]
        )
        for block in palette
    ]
    payload.extend(
        _named_tag(9, "palette", _list_payload(10, palette_payloads))
    )
    block_payloads = []
    for position, state in blocks:
        block_payloads.append(
            _compound_payload(
                [
                    _named_tag(
                        9,
                        "pos",
                        _list_payload(
                            3,
                            [_int_payload(value) for value in position],
                        ),
                    ),
                    _named_tag(3, "state", _int_payload(state)),
                ]
            )
        )
    payload.extend(_named_tag(9, "blocks", _list_payload(10, block_payloads)))
    payload.extend(_named_tag(9, "entities", _list_payload(10, [])))
    payload.extend(b"\x00")
    root = bytes([10]) + _string_payload("") + bytes(payload)
    return gzip.compress(root, mtime=0)


def _named_tag(tag_type: int, name: str, payload: bytes) -> bytes:
    return bytes([tag_type]) + _string_payload(name) + payload


def _compound_payload(tags: Iterable[bytes]) -> bytes:
    return b"".join(tags) + b"\x00"


def _list_payload(tag_type: int, payloads: list[bytes]) -> bytes:
    return bytes([tag_type]) + struct.pack(">i", len(payloads)) + b"".join(payloads)


def _int_payload(value: int) -> bytes:
    return struct.pack(">i", int(value))


def _string_payload(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > 65535:
        raise WorldCompileError("NBT string is too long.")
    return struct.pack(">H", len(encoded)) + encoded


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
