from __future__ import annotations

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


def generate_assets(
    router: ModelRouter,
    proposal: CompleteProposal,
    project_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CompleteProductionError(
            "Pillow is required for texture post-processing."
        ) from exc
    generated: list[dict[str, Any]] = []
    concept_dir = run_root / "asset-concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    for index, request in enumerate(proposal.assets):
        concept = router.generate_image(
            "image_generator",
            prompt=(
                "Minecraft Java texture source, centered, clean silhouette, no text, no watermark. "
                + request.prompt
            ),
            output_path=concept_dir / f"{request.asset_id}.png",
            width=min(1024, max(512, request.width)),
            height=min(1024, max(512, request.height)),
            seed=index,
        )
        target = (project_root / request.target_path).resolve()
        try:
            target.relative_to(project_root)
        except ValueError as exc:
            raise CompleteProductionError(
                "Asset target escaped the project root."
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(concept) as image:
            image.convert("RGBA").resize(
                (request.width, request.height), Image.Resampling.NEAREST
            ).save(target)
        generated.append(
            {
                "asset_id": request.asset_id,
                "concept": str(concept),
                "target": str(target),
                "sha256": sha256_file(target),
            }
        )
    return {
        "schema_version": "mmm/complete-assets-v2",
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
    client = BlockbenchMCPClient()
    try:
        client.call("open_project", {"path": geo})
        uv = client.call("validate_uv", {})
        render = client.call(
            "render_preview", {"output_path": str(preview)}
        )
        client.call("close_project", {})
    finally:
        client.close()
    return {
        "entity": gecko_receipt["entity_id"],
        "uv": uv,
        "render": render,
        "preview": str(preview),
    }


def run_playtest(actions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    bridge = MineflayerBridge()
    results: list[dict[str, Any]] = []
    allowed = MineflayerBridge.ACTIONS - {"connect", "disconnect"}
    try:
        results.append(
            bridge.call(
                "connect",
                host="127.0.0.1",
                port=25565,
                username="MMMTestBot",
            )
        )
        for action in actions:
            if not isinstance(action, dict) or "action" not in action:
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
            results.append(bridge.call(name, **params))
        results.append(bridge.call("inventory"))
        return {
            "schema_version": "mmm/playtest-result-v2",
            "status": "PASS",
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
                    "Return JSON {status: PASS|FAIL, findings: [...], acceptance_test_results: [...]} "
                    "for Minecraft runtime screenshots. Reject missing textures, broken models, unreadable GUI, "
                    "animation clipping and deviations from the approved design."
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
    if value.get("status") not in {"PASS", "FAIL"} or not isinstance(
        value.get("findings"), list
    ):
        raise CompleteProductionError(
            "VisualCritic returned an invalid result contract."
        )
    return {
        "schema_version": "mmm/visual-review-v1",
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
            "complete-proposal.json",
            json.dumps(
                proposal.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    return str(target)


def runtime_profile(run_root: Path, memory_mb: int) -> Path:
    path = run_root / "integration-inputs/runtime-profile.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmm/runtime-profiles-v1",
        "profiles": {
            "fabric_1201_disposable": {
                "minecraft_version": "1.20.1",
                "java_project_version": 17,
                "server_java_command": "java",
                "server_memory_mb": memory_mb,
                "server_launcher_relative": "runtime/fabric-server-launch.jar",
                "client_command_env": "MMM_MINECRAFT_CLIENT_COMMAND_JSON",
                "allowed_server_commands": [
                    "^list$",
                    "^stop$",
                    "^say [A-Za-z0-9 _.,!?-]{1,120}$",
                    "^gametest runall$",
                    "^tp testplayer -?[0-9]{1,7} -?[0-9]{1,7} -?[0-9]{1,7}$",
                    "^give testplayer [a-z0-9_.-]+:[a-z0-9_./-]+( [1-9][0-9]{0,3})?$",
                ],
                "startup_ready_patterns": [
                    "Done \\([0-9.]+s\\)! For help, type",
                    "For help, type \\\"help\\\"",
                ],
                "disposable_only": True,
                "eula_must_be_explicitly_accepted": True,
            }
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
