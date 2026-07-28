from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .capability_plugins import plugin_manifest
from .capabilities import capability_manifest
from .mcp_tools import MMMToolService
from .model_registry import ModelRegistry


mcp = FastMCP("M.M.M Minecraft Mod AI")


def _service() -> MMMToolService:
    return MMMToolService(
        workspace_root=os.environ.get("MMM_WORKSPACE", "mmm-output"),
        profile=os.environ.get("MMM_MODEL_PROFILE", "t4_local"),
    )


@mcp.tool()
def plan_game(prompt: str, media_paths: list[str] | None = None) -> dict[str, Any]:
    """Create a multimodal game design and an honest buildable Fabric slice."""
    return _service().plan_game(prompt, media_paths or [])


@mcp.tool()
def revise_plan(
    original_prompt: str,
    revision: str,
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Re-plan from the original brief plus an explicit user revision."""
    return _service().revise_plan(original_prompt, revision, media_paths or [])


@mcp.tool()
def approve_plan(proposal: dict[str, Any], approval_hash: str) -> dict[str, Any]:
    """Approve exactly the immutable proposal hash supplied by the user."""
    return _service().approve_plan(proposal, approval_hash)


@mcp.tool()
def search_project_rag(
    query: str,
    minecraft_version: str = "1.20.1",
    limit: int = 6,
) -> dict[str, Any]:
    """Search the code-owned, version-pinned Fabric evidence catalog."""
    return _service().search_project_rag(query, minecraft_version, limit)


@mcp.tool()
def inspect_existing_mod(archive_path: str) -> dict[str, Any]:
    """Inspect a source or release ZIP without executing its contents."""
    return _service().inspect_existing_mod(archive_path)


@mcp.tool()
def generate_fabric_project(
    proposal: dict[str, Any],
    approval_hash: str,
    run_name: str = "mcp-run",
) -> dict[str, Any]:
    """Generate the currently implemented Fabric source slice after approval."""
    return _service().generate_fabric_project(proposal, approval_hash, run_name)


@mcp.tool()
def generate_assets(
    assets: dict[str, str],
    output_dir: str = "assets-generated",
    seed: int = 0,
) -> dict[str, Any]:
    """Generate concept art and 16x16 Minecraft texture candidates on an exclusive GPU."""
    return _service().generate_assets(assets, output_dir, seed)


@mcp.tool()
def generate_world_ir(
    brief: str,
    output_path: str = "world/world-ir.json",
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Generate validated world-planning IR; this does not claim to compile world files."""
    return _service().generate_world_ir(brief, output_path, media_paths or [])


@mcp.tool()
def run_static_validation(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run deterministic source/resource validation inside the approved workspace."""
    return _service().run_static_validation(project_root, proposal, approval_hash)


@mcp.tool()
def run_gradle_build(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run the pinned Gradle clean build after approval."""
    return _service().run_gradle_build(project_root, proposal, approval_hash)


@mcp.tool()
def run_gametest(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
) -> dict[str, Any]:
    """Run the pinned Fabric headless GameTest server after approval."""
    return _service().run_gametest(project_root, proposal, approval_hash)


@mcp.tool()
def inspect_jar(jar_path: str, proposal: dict[str, Any]) -> dict[str, Any]:
    """Inspect a built JAR and verify its required Fabric contents."""
    return _service().inspect_jar(jar_path, proposal)


@mcp.tool()
def package_release(
    project_root: str,
    proposal: dict[str, Any],
    approval_hash: str,
    output_zip: str = "releases/mmm-release.zip",
    jar_path: str | None = None,
) -> dict[str, Any]:
    """Package validated source and an optional independently validated JAR."""
    return _service().package_release(
        project_root, proposal, approval_hash, output_zip, jar_path
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
    path = Path(__file__).resolve().parents[1] / "config" / "agent_roles.yaml"
    return path.read_text(encoding="utf-8")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
