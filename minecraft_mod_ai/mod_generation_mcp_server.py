from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from mcp.server import MCPServer

from .mcp_tools import MMMToolService
from .mod_development_methods import (
    mod_development_method_catalog,
    resolve_mod_development_methods,
)
from .production_tools import ProductionToolService

mcp = MCPServer("M.M.M Mod Generation", version="0.8.0")


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
def discover_mod_generation_capabilities() -> dict[str, Any]:
    """Return the mod-only generation surface and its explicit map exclusion."""
    return {
        "schema_version": "mmm/mod-generation-capabilities-v1",
        "scope": "MINECRAFT_FABRIC_MOD_PROJECT",
        "standalone_map_generation": False,
        "tools": [
            "resolve_mod_methods",
            "inspect_existing_mod",
            "search_code_rag",
            "generate_fabric_project",
            "generate_assets",
            "generate_geckolib_entity",
            "generate_system_plugin",
            "apply_source_patch",
        ],
        "worldgen_policy": (
            "Structures, biomes, dimensions and configured/placed features are "
            "ordinary mod modules only when explicitly requested. World saves, map "
            "ZIPs, schematics and external Builder block-delta handoffs are excluded."
        ),
    }


@mcp.resource("mmm://mod-development/methods")
def mod_method_catalog_resource() -> dict[str, Any]:
    return mod_development_method_catalog()


@mcp.tool()
def resolve_mod_methods(
    request: str,
    existing_project: bool = False,
) -> dict[str, Any]:
    """Resolve the implementation methods and gates required by one mod request."""
    return resolve_mod_development_methods(
        request,
        existing_project=existing_project,
    )


@mcp.tool()
def inspect_existing_mod(archive_path: str) -> dict[str, Any]:
    """Inspect an owned source/release ZIP without executing its contents."""
    return _core().inspect_existing_mod(archive_path)


@mcp.tool()
def search_code_rag(
    query: str,
    index_path: str = "rag/project-index.json",
    limit: int = 8,
    semantic: bool = False,
    rerank: bool = False,
    required_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search version-pinned project/API code before generation or patching."""
    return _production().search_code_rag(
        query,
        index_path=index_path,
        limit=limit,
        semantic=semantic,
        rerank=rerank,
        required_metadata=required_metadata,
    )


@mcp.tool()
def generate_fabric_project(
    proposal: dict[str, Any],
    approval_hash: str,
    run_name: str = "mcp-run",
) -> dict[str, Any]:
    """Generate the approved Fabric 1.20.1 mod project without a map artifact."""
    return _core().generate_fabric_project(proposal, approval_hash, run_name)


@mcp.tool()
def generate_assets(
    assets: dict[str, str],
    output_dir: str = "assets-generated",
    seed: int = 0,
) -> dict[str, Any]:
    """Generate requested mod textures and concept assets."""
    return _core().generate_assets(assets, output_dir, seed)


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
    """Generate entity, renderer, model and animation bindings."""
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
    """Generate requested gameplay, persistence, networking or multiplayer systems."""
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
def apply_source_patch(
    project_root: str,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply a transactional SHA-256 guarded source patch with rollback."""
    return _core().apply_source_patch(project_root, operations)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
