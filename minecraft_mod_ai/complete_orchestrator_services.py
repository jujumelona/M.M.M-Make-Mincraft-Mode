from __future__ import annotations

from .fixed_template_generation import generate_fixed_template_text
from .model_response_templates import response_schema

import json
import os
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .blockbench_client import BlockbenchMCPClient
from .complete_orchestrator_support import CompleteProductionError, _extract_json
from .complete_spec import CompleteProposal
from .mineflayer_bridge import MineflayerBridge
from .model_router import ModelRouter
from .task_template_catalog import load_template


def generate_assets(router: ModelRouter, proposal: CompleteProposal, project_root: Path, run_root: Path) -> dict[str, Any]:
    """Single canonical resource-asset entrypoint."""
    from .resource_asset_production import generate_assets as canonical_generate_assets
    return canonical_generate_assets(router, proposal, project_root, run_root)


generate_assets._mmm_adaptive_image_gpu_session = True  # type: ignore[attr-defined]


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


def _visual_review_instruction() -> tuple[str, str]:
    """Load the semantic visual-review policy from the runtime template authority."""
    template = load_template("validation/visual_review")
    declared_input = template.get("input")
    expected_input = {"game_design", "acceptance_tests", "runtime_screenshots"}
    if not isinstance(declared_input, dict) or set(declared_input) != expected_input:
        raise CompleteProductionError(
            "Visual-review template input contract does not match the production consumer."
        )
    task = str(template.get("task") or "").strip()
    rules = template.get("rules")
    if not task or not isinstance(rules, list) or not rules or any(
        not str(rule).strip() for rule in rules
    ):
        raise CompleteProductionError(
            "Visual-review template must provide one task and explicit non-empty rules."
        )
    response_contract = str(template.get("response_contract") or "").strip()
    if not response_contract:
        raise CompleteProductionError(
            "Visual-review template must name its fixed response contract."
        )
    # Resolve now so an invalid contract fails before the multimodal model call.
    response_schema(response_contract)
    instruction = task + "\nRules:\n" + "\n".join(
        f"- {str(rule).strip()}" for rule in rules
    )
    return instruction, response_contract


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
    instruction, response_contract = _visual_review_instruction()
    text = generate_fixed_template_text(router,
        "visual_critic",
        [
            {
                "role": "system",
                "content": instruction,
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
        response_schema=response_schema(response_contract),
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
