from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .blockbench_client import BlockbenchMCPClient
from .complete_spec import CompleteProposal
from .complete_orchestrator_support import CompleteProductionError, _extract_json
from .mineflayer_bridge import MineflayerBridge
from .model_router import ModelRouter
from .source_patch import sha256_file


_MODEL_TILE_MIN = 256
_MODEL_TILE_MAX = 1024
_MODEL_TILE_ALIGNMENT = 16
_SOURCE_TILE_OVERLAP = 128


def _aligned_model_dimension(value: int) -> int:
    bounded = min(_MODEL_TILE_MAX, max(_MODEL_TILE_MIN, value))
    return min(
        _MODEL_TILE_MAX,
        (
            (bounded + _MODEL_TILE_ALIGNMENT - 1)
            // _MODEL_TILE_ALIGNMENT
        )
        * _MODEL_TILE_ALIGNMENT,
    )


def _axis_source_tiles(length: int) -> tuple[tuple[int, int], ...]:
    """Return overlapping source-tile starts and model-safe dimensions."""

    starts = [0]
    stride = _MODEL_TILE_MAX - _SOURCE_TILE_OVERLAP
    while starts[-1] + _MODEL_TILE_MAX < length:
        next_start = starts[-1] + stride
        if length - next_start < _MODEL_TILE_MIN:
            next_start = length - _MODEL_TILE_MIN
        if next_start <= starts[-1]:
            raise CompleteProductionError(
                "Could not create a forward-only asset tile layout."
            )
        starts.append(next_start)
    return tuple(
        (
            start,
            _aligned_model_dimension(min(_MODEL_TILE_MAX, length - start)),
        )
        for start in starts
    )


def _overview_dimensions(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, _MODEL_TILE_MAX / max(width, height))
    return (
        _aligned_model_dimension(max(1, round(width * scale))),
        _aligned_model_dimension(max(1, round(height * scale))),
    )


def _stable_asset_seed(
    asset_id: str,
    layer: str,
    *,
    x: int = 0,
    y: int = 0,
) -> int:
    digest = hashlib.sha256(
        f"{asset_id}\0{layer}\0{x}\0{y}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _open_generated_rgba(
    Image: Any,
    path: Path,
    *,
    expected_size: tuple[int, int],
) -> Any:
    with Image.open(path) as source:
        source.load()
        if source.size != expected_size:
            raise CompleteProductionError(
                "Image generator returned an unexpected source-tile size: "
                f"{source.size!r}, expected {expected_size!r}."
            )
        return source.convert("RGBA")


def _palette_anchors(image: Any) -> str:
    rgb = image.convert("RGB")
    try:
        width, height = rgb.size
        anchors: list[str] = []
        for y_fraction in (1, 3, 5):
            for x_fraction in (1, 3, 5):
                red, green, blue = rgb.getpixel(
                    (
                        min(width - 1, width * x_fraction // 6),
                        min(height - 1, height * y_fraction // 6),
                    )
                )
                color = f"#{red:02x}{green:02x}{blue:02x}"
                if color not in anchors:
                    anchors.append(color)
        return ", ".join(anchors)
    finally:
        rgb.close()


def _feather_mask(
    Image: Any,
    ImageChops: Any,
    size: tuple[int, int],
    *,
    left_overlap: int,
    top_overlap: int,
) -> Any:
    width, height = size
    mask = Image.new("L", size, 255)
    if left_overlap:
        ramp = Image.new("L", (left_overlap, 1))
        denominator = max(1, left_overlap - 1)
        ramp.putdata(
            [round(255 * offset / denominator) for offset in range(left_overlap)]
        )
        mask.paste(ramp.resize((left_overlap, height)), (0, 0))
    if top_overlap:
        ramp = Image.new("L", (1, top_overlap))
        denominator = max(1, top_overlap - 1)
        ramp.putdata(
            [round(255 * offset / denominator) for offset in range(top_overlap)]
        )
        top_mask = Image.new("L", size, 255)
        top_mask.paste(ramp.resize((width, top_overlap)), (0, 0))
        mask = ImageChops.multiply(mask, top_mask)
    return mask


def _generate_single_asset_source(
    router: ModelRouter,
    Image: Any,
    *,
    request: Any,
    base_prompt: str,
    concept_dir: Path,
    target: Path,
) -> dict[str, Any]:
    source_size = (
        _aligned_model_dimension(request.width),
        _aligned_model_dimension(request.height),
    )
    concept = router.generate_image(
        "image_generator",
        prompt=base_prompt,
        output_path=concept_dir / f"{request.asset_id}.png",
        width=source_size[0],
        height=source_size[1],
        seed=_stable_asset_seed(request.asset_id, "single"),
    )
    image = _open_generated_rgba(
        Image,
        Path(concept),
        expected_size=source_size,
    )
    try:
        if image.size != (request.width, request.height):
            image = image.resize(
                (request.width, request.height),
                Image.Resampling.LANCZOS,
            )
        image.save(target)
    finally:
        image.close()
    return {
        "concept": str(concept),
        "source_mode": "single_source",
        "source_size": list(source_size),
        "source_tiles": [],
    }


def _generate_tiled_asset_source(
    router: ModelRouter,
    Image: Any,
    ImageChops: Any,
    *,
    request: Any,
    base_prompt: str,
    concept_dir: Path,
    target: Path,
) -> dict[str, Any]:
    asset_dir = concept_dir / request.asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    overview_size = _overview_dimensions(request.width, request.height)
    overview_seed = _stable_asset_seed(request.asset_id, "overview")
    overview = router.generate_image(
        "image_generator",
        prompt=(
            f"{base_prompt} Create the global composition and palette guide for "
            f"a {request.width}x{request.height} final source. Keep all edges "
            "and major forms consistent for later overlapping detail tiles."
        ),
        output_path=asset_dir / "overview.png",
        width=overview_size[0],
        height=overview_size[1],
        seed=overview_seed,
    )
    overview_image = _open_generated_rgba(
        Image,
        Path(overview),
        expected_size=overview_size,
    )
    try:
        palette_anchors = _palette_anchors(overview_image)
    finally:
        overview_image.close()

    x_tiles = _axis_source_tiles(request.width)
    y_tiles = _axis_source_tiles(request.height)
    canvas = Image.new(
        "RGBA",
        (request.width, request.height),
        (0, 0, 0, 0),
    )
    tile_receipts: list[dict[str, Any]] = []
    try:
        for row, (y, source_height) in enumerate(y_tiles):
            previous_y_end = (
                min(
                    request.height,
                    y_tiles[row - 1][0] + y_tiles[row - 1][1],
                )
                if row
                else y
            )
            for column, (x, source_width) in enumerate(x_tiles):
                previous_x_end = (
                    min(
                        request.width,
                        x_tiles[column - 1][0] + x_tiles[column - 1][1],
                    )
                    if column
                    else x
                )
                tile_seed = _stable_asset_seed(
                    request.asset_id,
                    "detail",
                    x=x,
                    y=y,
                )
                tile_path = router.generate_image(
                    "image_generator",
                    prompt=(
                        f"{base_prompt} Generate a native-resolution detail tile "
                        f"for final pixel region x={x}, y={y}, "
                        f"width={min(source_width, request.width - x)}, "
                        f"height={min(source_height, request.height - y)} of the "
                        f"{request.width}x{request.height} source. Match the global "
                        f"composition guide seed {overview_seed} and its sampled "
                        f"palette anchors {palette_anchors}. "
                        "Continue forms and colors seamlessly across every "
                        "overlapping edge."
                    ),
                    output_path=asset_dir / f"tile-y{y:06d}-x{x:06d}.png",
                    width=source_width,
                    height=source_height,
                    seed=tile_seed,
                )
                tile = _open_generated_rgba(
                    Image,
                    Path(tile_path),
                    expected_size=(source_width, source_height),
                )
                visible_width = min(source_width, request.width - x)
                visible_height = min(source_height, request.height - y)
                if tile.size != (visible_width, visible_height):
                    cropped = tile.crop(
                        (0, 0, visible_width, visible_height)
                    )
                    tile.close()
                    tile = cropped
                left_overlap = (
                    min(visible_width, max(0, previous_x_end - x))
                    if column
                    else 0
                )
                top_overlap = (
                    min(visible_height, max(0, previous_y_end - y))
                    if row
                    else 0
                )
                try:
                    mask = _feather_mask(
                        Image,
                        ImageChops,
                        tile.size,
                        left_overlap=left_overlap,
                        top_overlap=top_overlap,
                    )
                    try:
                        canvas.paste(tile, (x, y), mask)
                    finally:
                        mask.close()
                finally:
                    tile.close()
                tile_receipts.append(
                    {
                        "path": str(tile_path),
                        "x": x,
                        "y": y,
                        "source_width": source_width,
                        "source_height": source_height,
                        "visible_width": visible_width,
                        "visible_height": visible_height,
                        "left_overlap": left_overlap,
                        "top_overlap": top_overlap,
                        "seed": tile_seed,
                        "sha256": sha256_file(tile_path),
                    }
                )
        canvas.save(target)
    finally:
        canvas.close()
    return {
        "concept": str(overview),
        "source_mode": "multiscale_overlapping_tiles",
        "overview_size": list(overview_size),
        "overview_sha256": sha256_file(overview),
        "palette_anchors": palette_anchors.split(", "),
        "source_tiles": tile_receipts,
    }


def generate_assets(
    router: ModelRouter,
    proposal: CompleteProposal,
    project_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageChops
    except ImportError as exc:
        raise CompleteProductionError(
            "Pillow is required for texture post-processing."
        ) from exc
    generated: list[dict[str, Any]] = []
    concept_dir = run_root / "asset-concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    resolved_project_root = project_root.resolve()
    for request in proposal.assets:
        base_prompt = (
            "Minecraft Java texture source, centered, clean silhouette, no "
            f"text, no watermark. {request.prompt}"
        )
        target = (resolved_project_root / request.target_path).resolve()
        try:
            target.relative_to(resolved_project_root)
        except ValueError as exc:
            raise CompleteProductionError(
                "Asset target escaped the project root."
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            request.width > _MODEL_TILE_MAX
            or request.height > _MODEL_TILE_MAX
        ):
            source_receipt = _generate_tiled_asset_source(
                router,
                Image,
                ImageChops,
                request=request,
                base_prompt=base_prompt,
                concept_dir=concept_dir,
                target=target,
            )
        else:
            source_receipt = _generate_single_asset_source(
                router,
                Image,
                request=request,
                base_prompt=base_prompt,
                concept_dir=concept_dir,
                target=target,
            )
        with Image.open(target) as final_image:
            if final_image.size != (request.width, request.height):
                raise CompleteProductionError(
                    "Asset composition did not preserve the requested dimensions."
                )
        generated.append(
            {
                "asset_id": request.asset_id,
                "target": str(target),
                "width": request.width,
                "height": request.height,
                "sha256": sha256_file(target),
                **source_receipt,
            }
        )
    return {
        "schema_version": "mmm/complete-assets-v3",
        "status": "GENERATED",
        "assets": generated,
    }


def blockbench_review(
    gecko_receipt: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    geo = next(
        (
            path
            for path in gecko_receipt.get("files", [])
            if str(path).endswith(".geo.json")
        ),
        None,
    )
    if not geo:
        raise CompleteProductionError(
            "GeckoLib receipt did not contain geometry."
        )
    preview = run_root / "blockbench-previews" / (Path(geo).stem + ".png")
    preview.parent.mkdir(parents=True, exist_ok=True)
    client = BlockbenchMCPClient(workspace_root=run_root)
    try:
        client.call("open_project", {"path": geo})
        uv = client.call("validate_uv", {})
        render = client.call(
            "render_preview", {"output_path": str(preview)}
        )
        client.call("close_project", {})
    finally:
        client.close()
    if not isinstance(uv, dict) or uv.get("status") not in {"PASS", "OK"}:
        raise CompleteProductionError(
            "Blockbench UV validation did not return a passing receipt."
        )
    if not preview.is_file() or preview.is_symlink():
        raise CompleteProductionError(
            "Blockbench did not produce a regular preview image."
        )
    return {
        "entity": gecko_receipt["entity_id"],
        "uv": uv,
        "render": render,
        "preview": str(preview),
    }


def run_playtest(actions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    requested = tuple(actions)
    if not requested:
        raise CompleteProductionError(
            "Complete runtime verification requires explicit playtest actions; an empty bot session cannot prove functionality."
        )
    allowed = MineflayerBridge.ACTIONS - {"connect", "disconnect"}
    observational = {"status", "inventory"}
    normalized: list[tuple[str, dict[str, Any]]] = []
    has_interaction = False
    has_assertion = False
    for action in requested:
        if not isinstance(action, dict) or set(action) - {"action", "params"}:
            raise CompleteProductionError(
                "Every playtest action must contain only action and optional params."
            )
        if "action" not in action:
            raise CompleteProductionError(
                "Every playtest action must contain action."
            )
        name = str(action["action"])
        if name not in allowed:
            raise CompleteProductionError(
                f"Unsupported playtest action: {name}"
            )
        params = action.get("params", {})
        if not isinstance(params, dict):
            raise CompleteProductionError(
                "Playtest params must be an object."
            )
        if name not in observational and name != "wait_for":
            has_interaction = True
        if name == "wait_for":
            has_assertion = True
        normalized.append((name, params))
    if not has_interaction:
        raise CompleteProductionError(
            "Complete playtesting must perform at least one gameplay interaction, not only status or inventory reads."
        )
    if not has_assertion:
        raise CompleteProductionError(
            "Complete playtesting must include wait_for so the requested outcome is machine-checked."
        )

    bridge = MineflayerBridge()
    results: list[dict[str, Any]] = []
    try:
        results.append(
            bridge.call(
                "connect",
                host="127.0.0.1",
                port=25565,
                username="MMMTestBot",
            )
        )
        for name, params in normalized:
            result = bridge.call(name, **params)
            if name == "wait_for" and result.get("matched") is not True:
                raise CompleteProductionError(
                    "Mineflayer wait_for returned without a matched condition."
                )
            results.append(
                {
                    "action": name,
                    "params": params,
                    "result": result,
                }
            )
        results.append(
            {
                "action": "inventory",
                "result": bridge.call("inventory"),
            }
        )
        return {
            "schema_version": "mmm/playtest-result-v3",
            "status": "PASS",
            "interaction_count": sum(
                1
                for name, _ in normalized
                if name not in observational and name != "wait_for"
            ),
            "assertion_count": sum(
                1 for name, _ in normalized if name == "wait_for"
            ),
            "results": results,
        }
    finally:
        bridge.close()


def visual_review(
    router: ModelRouter,
    proposal: CompleteProposal,
    screenshots: tuple[str, ...],
) -> dict[str, Any]:
    paths = [Path(value).expanduser().resolve() for value in screenshots]
    if not paths:
        raise CompleteProductionError(
            "Visual review requires at least one runtime screenshot."
        )
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise CompleteProductionError(
            "Every visual-review screenshot must be a regular file."
        )
    text = router.generate_text(
        "visual_critic",
        [
            {
                "role": "system",
                "content": (
                    "Return JSON {status: PASS|FAIL, findings: [...], acceptance_test_results: "
                    "[{test: string, status: PASS|FAIL, evidence: string}]}. Return exactly one result "
                    "for every supplied acceptance test. Reject missing textures, broken models, unreadable GUI, "
                    "animation clipping and deviations from the approved design. Do not mark non-visual behavior "
                    "as PASS unless the screenshot visibly proves it."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "game_design": proposal.game_design,
                        "acceptance_tests": list(proposal.acceptance_tests),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        media_paths=paths,
        response_format="json",
    )
    value = _extract_json(text)
    if set(value) != {
        "status",
        "findings",
        "acceptance_test_results",
    }:
        raise CompleteProductionError(
            "VisualCritic returned invalid top-level fields."
        )
    findings = value["findings"]
    test_results = value["acceptance_test_results"]
    if value["status"] not in {"PASS", "FAIL"} or not isinstance(
        findings, list
    ) or not isinstance(test_results, list):
        raise CompleteProductionError(
            "VisualCritic returned an invalid result contract."
        )
    if len(test_results) != len(proposal.acceptance_tests):
        raise CompleteProductionError(
            "VisualCritic did not return one result per acceptance test."
        )
    expected = list(proposal.acceptance_tests)
    for index, result in enumerate(test_results):
        if not isinstance(result, dict) or set(result) != {
            "test",
            "status",
            "evidence",
        }:
            raise CompleteProductionError(
                "VisualCritic acceptance result fields are invalid."
            )
        if str(result["test"]) != expected[index]:
            raise CompleteProductionError(
                "VisualCritic acceptance results changed or reordered the approved tests."
            )
        if result["status"] not in {"PASS", "FAIL"} or not str(
            result["evidence"]
        ).strip():
            raise CompleteProductionError(
                "VisualCritic acceptance result lacks a valid status or evidence."
            )
    if value["status"] == "PASS" and any(
        result["status"] != "PASS" for result in test_results
    ):
        raise CompleteProductionError(
            "VisualCritic overall PASS conflicts with a failed acceptance test."
        )
    return {
        "schema_version": "mmm/visual-review-v2",
        **value,
        "screenshots": [str(path) for path in paths],
    }


def package_source_only(
    run_root: Path,
    project_root: Path,
    proposal: CompleteProposal,
) -> str:
    target = run_root / "releases/complete-source.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(project_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(project_root)
            if any(
                part in {".gradle", "build", "run", ".cache"}
                for part in relative.parts
            ):
                continue
            archive.write(path, Path("source") / relative)
        archive.writestr(
            "complete-proposal-location.json",
            json.dumps(
                {
                    "schema_version": "mmm/complete-proposal-location-v1",
                    "path": "source/.minecraft_ai/complete-proposal.json",
                    "proposal_hash": proposal.calculate_hash(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    return str(target)


def runtime_profile(run_root: Path, memory_mb: int) -> Path:
    version = os.environ.get("MMM_MINECRAFT_VERSION", "").strip()
    loader = os.environ.get("MMM_LOADER", "").strip().casefold()
    java_raw = os.environ.get("MMM_JAVA_VERSION", "").strip()
    if not version or not loader or not java_raw:
        raise CompleteProductionError(
            "Runtime profile requires an explicit approved Minecraft target; "
            "the platform runtime contract normally supplies it."
        )
    try:
        java_version = int(java_raw)
    except ValueError as exc:
        raise CompleteProductionError("MMM_JAVA_VERSION must be an integer.") from exc
    path = run_root / "integration-inputs/runtime-profile.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmm/runtime-profiles-v1",
        "profiles": {
            "minecraft_target_disposable": {
                "minecraft_version": version,
                "loader": loader,
                "java_project_version": java_version,
                "server_java_command": "java",
                "server_memory_mb": memory_mb,
                "server_launcher_relative": "runtime/fabric-server-launch.jar",
                "client_command_env": "MMM_MINECRAFT_CLIENT_COMMAND_JSON",
                "allowed_server_commands": [
                    "^list$", "^stop$", "^say [A-Za-z0-9 _.,!?-]{1,120}$",
                    "^gametest runall$",
                    "^tp testplayer -?[0-9]{1,7} -?[0-9]{1,7} -?[0-9]{1,7}$",
                    "^give testplayer [a-z0-9_.-]+:[a-z0-9_./-]+( [1-9][0-9]{0,3})?$",
                ],
                "startup_ready_patterns": [
                    "Done \\([0-9.]+s\\)! For help, type",
                    "For help, type \"help\"",
                ],
                "disposable_only": True,
                "eula_must_be_explicitly_accepted": True,
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
