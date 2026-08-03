from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from mcp.server import MCPServer

from .builder_contract_service import BuilderContractService

mcp = MCPServer("M.M.M Builder Contract", version="0.7.0")


@lru_cache(maxsize=1)
def _service() -> BuilderContractService:
    return BuilderContractService(
        workspace_root=os.environ.get("MMM_WORKSPACE", "mmm-output"),
        profile=os.environ.get("MMM_MODEL_PROFILE", "t4_local"),
    )


@mcp.tool()
def discover_builder_contract() -> dict[str, Any]:
    """Return the strict central-agent to external-Builder boundary."""
    return _service().contract()


@mcp.tool()
def search_buildspec_rag(
    query: str,
    limit: int = 12,
) -> dict[str, Any]:
    """Retrieve central-only architecture ontology and operator evidence."""
    return _service().search_buildspec_rag(query, limit)


@mcp.tool()
def plan_architecture_buildspec(
    request: str,
    world: dict[str, Any],
    media_paths: list[str] | None = None,
    external_evidence: list[dict[str, Any]] | None = None,
    rag_limit: int = 12,
) -> dict[str, Any]:
    """Use the central VLM/RAG agent to emit validated buildspec_v2 only."""
    return _service().plan_buildspec(
        request,
        world,
        media_paths or [],
        external_evidence or [],
        rag_limit,
    )


@mcp.tool()
def validate_architecture_buildspec(
    buildspec: dict[str, Any],
) -> dict[str, Any]:
    """Reject natural-language leakage and invalid Builder references."""
    return _service().validate_buildspec(buildspec)


@mcp.tool()
def prepare_external_builder_handoff(
    buildspec: dict[str, Any],
    handoff_dir: str = "builder-handoff",
    require_world_artifacts: bool = True,
) -> dict[str, Any]:
    """Write canonical buildspec.json without pretending to execute Builder."""
    return _service().prepare_builder_handoff(
        buildspec,
        handoff_dir,
        require_world_artifacts,
    )


@mcp.tool()
def validate_external_builder_result(
    buildspec: dict[str, Any],
    result: dict[str, Any],
    result_dir: str = "",
    require_result_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate Builder block-delta refs, port partition and predictions."""
    return _service().validate_builder_result(
        buildspec,
        result,
        result_dir,
        require_result_artifacts,
    )


@mcp.resource("mmm://builder/buildspec-v2")
def buildspec_resource() -> str:
    return json.dumps(
        _service().contract(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


@mcp.resource("mmm://builder/catalog")
def builder_catalog_resource() -> str:
    return _service().catalog.path.read_text(encoding="utf-8")


@mcp.prompt()
def central_agent_to_builder(request: str) -> str:
    """Keep natural-language interpretation on the central-agent side."""
    return (
        "Interpret this request with the central VLM and architecture RAG. "
        "Convert it to buildspec_v2. Never forward the request, captions, style "
        "sentences or retrieved prose to Builder. Builder receives only validated "
        f"machine geometry and constraints. Request: {request}"
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
