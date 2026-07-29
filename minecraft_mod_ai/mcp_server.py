from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .capability_plugins import plugin_manifest
from .config_paths import config_path
from .capabilities import capability_manifest
from .external_mcp import ExternalMCPRegistry
from .mcp_tools import MMMToolService
from .model_registry import ModelRegistry
from .production_tools import ProductionToolService
from .skill_catalog import validate_skill_catalog


mcp = FastMCP("M.M.M Minecraft Mod AI")


@lru_cache(maxsize=1)
def _core() -> MMMToolService:
    return MMMToolService(
        workspace_root=os.environ.get("MMM_WORKSPACE", "mmm-output"),
        profile=os.environ.get("MMM_MODEL_PROFILE", "t4_local"),
    )


@lru_cache(maxsize=1)
def _production() -> ProductionToolService:
    return ProductionToolService(
        workspace_root=os.environ.get("MMM_WORKSPACE", "mmm-output"),
        profile=os.environ.get("MMM_MODEL_PROFILE", "t4_local"),
    )


@mcp.tool()
def plan_game(prompt: str, media_paths: list[str] | None = None) -> dict[str, Any]:
    """Create a multimodal game design and an honest buildable Fabric slice."""
    return _core().plan_game(prompt, media_paths or [])


@mcp.tool()
def revise_plan(
    original_prompt: str,
    revision: str,
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Re-plan from the original brief plus an explicit user revision."""
    return _core().revise_plan(original_prompt, revision, media_paths or [])


@mcp.tool()
def approve_plan(proposal: dict[str, Any], approval_hash: str) -> dict[str, Any]:
    """Approve exactly the immutable proposal hash supplied by the user."""
    return _core().approve_plan(proposal, approval_hash)


@mcp.tool()
def search_project_rag(
    query: str,
    minecraft_version: str = "1.20.1",
    limit: int = 6,
) -> dict[str, Any]:
    """Search the code-owned, version-pinned primary evidence catalog."""
    return _core().search_project_rag(query, minecraft_version, limit)


@mcp.tool()
def index_project_rag(
    roots: list[str],
    metadata: dict[str, Any],
    index_path: str = "rag/project-index.json",
    semantic: bool = False,
) -> dict[str, Any]:
    """Build a version/license-aware project code index."""
    return _production().index_project_rag(
        roots,
        index_path=index_path,
        metadata=metadata,
        semantic=semantic,
    )


@mcp.tool()
def search_code_rag(
    query: str,
    index_path: str = "rag/project-index.json",
    limit: int = 8,
    semantic: bool = False,
    rerank: bool = False,
    required_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search project/API code with lexical, embedding and optional reranker stages."""
    return _production().search_code_rag(
        query,
        index_path=index_path,
        limit=limit,
        semantic=semantic,
        rerank=rerank,
        required_metadata=required_metadata,
    )


@mcp.tool()
def inspect_existing_mod(archive_path: str) -> dict[str, Any]:
    """Inspect a source or release ZIP without executing its contents."""
    return _core().inspect_existing_mod(archive_path)


@mcp.tool()
def generate_fabric_project(
    proposal: dict[str, Any],
    approval_hash: str,
    run_name: str = "mcp-run",
) -> dict[str, Any]:
    """Generate the core Fabric 1.20.1 project after immutable approval."""
    return _core().generate_fabric_project(proposal, approval_hash, run_name)


@mcp.tool()
def generate_assets(
    assets: dict[str, str],
    output_dir: str = "assets-generated",
    seed: int = 0,
) -> dict[str, Any]:
    """Generate concept art and Minecraft texture candidates on an exclusive GPU."""
    return _core().generate_assets(assets, output_dir, seed)


@mcp.tool()
def generate_world_ir(
    brief: str,
    output_path: str = "world/world-ir.json",
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Generate validated world-planning IR."""
    return _core().generate_world_ir(brief, output_path, media_paths or [])


@mcp.tool()
def compile_world_ir(
    world_ir: dict[str, Any],
    mod_id: str,
    output_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Compile WorldDesignIR into NBT, Jigsaw and 1.20.1 datapack resources."""
    return _production().compile_world(
        world_ir=world_ir,
        mod_id=mod_id,
        output_root=output_root,
        proposal=proposal,
        approval_hash=approval_hash,
    )


@mcp.tool()
def generate_geckolib_entity(
    project_root: str,
    mod_id: str,
    package_name: str,
    entity_id: str,
    proposal: dict[str, Any],
    approval_hash: str,
    geckolib_version: str = "4.8.2",
) -> dict[str, Any]:
    """Generate GeckoLib resources and build-gated binding contracts."""
    return _production().generate_geckolib_entity(
        project_root=project_root,
        mod_id=mod_id,
        package_name=package_name,
        entity_id=entity_id,
        proposal=proposal,
        approval_hash=approval_hash,
        geckolib_version=geckolib_version,
    )


@mcp.tool()
def generate_system_plugin(
    project_root: str,
    pack_id: str,
    mod_id: str,
    package_name: str,
    config: dict[str, Any],
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Generate quest/class/economy/networking/party domain foundations."""
    return _production().generate_system_plugin(
        project_root=project_root,
        pack_id=pack_id,
        mod_id=mod_id,
        package_name=package_name,
        config=config,
        proposal=proposal,
        approval_hash=approval_hash,
    )


@mcp.tool()
def java_diagnostics(
    project_root: str,
    relative_files: list[str] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Run real Eclipse JDT LS diagnostics for a generated Java project."""
    return _production().java_diagnostics(
        project_root,
        relative_files,
        timeout_seconds,
    )


@mcp.tool()
def java_workspace_symbols(
    project_root: str,
    query: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Search the real Java symbol graph through Eclipse JDT LS."""
    return _production().java_workspace_symbols(
        project_root,
        query,
        timeout_seconds,
    )


@mcp.tool()
def blockbench_list_tools() -> dict[str, Any]:
    """List only reviewed Blockbench MCP operations."""
    return _production().blockbench_list_tools()


@mcp.tool()
def blockbench_execute(
    operation: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute one reviewed Blockbench modeling operation."""
    return _production().blockbench_execute(operation, arguments)


@mcp.tool()
def run_static_validation(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run deterministic source/resource validation inside the approved workspace."""
    return _core().run_static_validation(project_root, proposal, approval_hash)


@mcp.tool()
def run_gradle_build(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run the pinned Gradle clean build after approval."""
    return _core().run_gradle_build(project_root, proposal, approval_hash)


@mcp.tool()
def run_gametest(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run the pinned Fabric headless GameTest server after approval."""
    return _core().run_gametest(project_root, proposal, approval_hash)


@mcp.tool()
def inspect_jar(jar_path: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Inspect a built JAR and verify its required Fabric contents."""
    return _core().inspect_jar(jar_path, proposal)


@mcp.tool()
def runtime_prepare_instance(
    instance_name: str,
    mod_jar: str,
    server_launcher: str,
    eula_accepted: bool,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Prepare a disposable local Minecraft 1.20.1 integration instance."""
    return _production().runtime_prepare_instance(
        instance_name=instance_name,
        mod_jar=mod_jar,
        server_launcher=server_launcher,
        eula_accepted=eula_accepted,
        proposal=proposal,
        approval_hash=approval_hash,
    )


@mcp.tool()
def runtime_start_server(timeout_seconds: int = 180) -> dict[str, Any]:
    """Start the configured disposable Fabric server and wait for readiness."""
    return _production().runtime_start_server(timeout_seconds)


@mcp.tool()
def runtime_start_client() -> dict[str, Any]:
    """Start the operator-configured client command in the disposable instance."""
    return _production().runtime_start_client()


@mcp.tool()
def runtime_send_command(command: str) -> dict[str, Any]:
    """Send only an allowlisted test-server command."""
    return _production().runtime_send_command(command)


@mcp.tool()
def runtime_logs(lines: int = 120) -> dict[str, Any]:
    """Read bounded server and client log tails."""
    return _production().runtime_logs(lines)


@mcp.tool()
def runtime_register_screenshot(screenshot_path: str) -> dict[str, Any]:
    """Register a screenshot produced inside the disposable workspace."""
    return _production().runtime_register_screenshot(screenshot_path)


@mcp.tool()
def runtime_status() -> dict[str, Any]:
    """Return current disposable runtime state."""
    return _production().runtime_status()


@mcp.tool()
def runtime_stop(cleanup: bool = False) -> dict[str, Any]:
    """Stop client/server and optionally delete the disposable instance."""
    return _production().runtime_stop(cleanup)


@mcp.tool()
def mineflayer_connect(
    host: str = "127.0.0.1",
    port: int = 25565,
    username: str = "MMMTestBot",
) -> dict[str, Any]:
    """Connect the first-party Mineflayer bridge to localhost Minecraft 1.20.1."""
    return _production().mineflayer_connect(host, port, username)


@mcp.tool()
def mineflayer_status() -> dict[str, Any]:
    """Return Mineflayer player state."""
    return _production().mineflayer_status()


@mcp.tool()
def mineflayer_walk_to(
    x: float,
    y: float,
    z: float,
    range: int = 1,
) -> dict[str, Any]:
    """Walk the test bot to a bounded target."""
    return _production().mineflayer_walk_to(x, y, z, range)


@mcp.tool()
def mineflayer_interact_block(x: int, y: int, z: int) -> dict[str, Any]:
    """Interact with one loaded block at explicit coordinates."""
    return _production().mineflayer_interact_block(x, y, z)


@mcp.tool()
def mineflayer_inventory() -> dict[str, Any]:
    """Read the test bot inventory."""
    return _production().mineflayer_inventory()


@mcp.tool()
def mineflayer_disconnect() -> dict[str, Any]:
    """Disconnect the test bot."""
    return _production().mineflayer_disconnect()


@mcp.tool()
def run_model_smoke(
    role: str,
    output_dir: str = "model-smoke",
    media_path: str | None = None,
    audio_path: str | None = None,
) -> dict[str, Any]:
    """Actually load and exercise one configured role and record VRAM/timing."""
    return _production().run_model_smoke(
        role,
        output_dir,
        media_path,
        audio_path,
    )


@mcp.tool()
def record_training_trace(
    trace: dict[str, Any],
    store_path: str = "training/traces",
) -> dict[str, Any]:
    """Record only a licensed, build/GameTest/JAR-verified training trace."""
    return _production().record_training_trace(trace, store_path)


@mcp.tool()
def export_training_dataset(
    store_path: str = "training/traces",
    output_path: str = "training/mmm-fabric-coder-1201.jsonl",
) -> dict[str, Any]:
    """Export verified traces to chat-format SFT JSONL."""
    return _production().export_training_dataset(store_path, output_path)


@mcp.tool()
def package_release(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
    output_zip: str = "releases/mmm-release.zip",
    jar_path: str | None = None,
) -> dict[str, Any]:
    """Package validated source and an optional independently validated JAR."""
    return _core().package_release(
        project_root,
        proposal,
        approval_hash,
        output_zip,
        jar_path,
    )


@mcp.resource("mmm://model-registry")
def model_registry_resource() -> str:
    return json.dumps(ModelRegistry().to_public_dict(), ensure_ascii=False, indent=2)


@mcp.resource("mmm://capabilities")
def capability_resource() -> str:
    return json.dumps(capability_manifest(), ensure_ascii=False, indent=2)


@mcp.resource("mmm://plugins")
def plugin_resource() -> str:
    return json.dumps(plugin_manifest(), ensure_ascii=False, indent=2)


@mcp.resource("mmm://agent-roles")
def agent_roles_resource() -> str:
    path = config_path("agent_roles.yaml")
    return path.read_text(encoding="utf-8")


@mcp.resource("mmm://external-mcp-registry")
def external_mcp_registry_resource() -> str:
    return json.dumps(
        ExternalMCPRegistry().public_dict(),
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("mmm://skill-catalog")
def skill_catalog_resource() -> str:
    return json.dumps(validate_skill_catalog(), ensure_ascii=False, indent=2)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
