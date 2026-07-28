from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .spec import canonical_json


@dataclass(frozen=True)
class CapabilityRecord:
    name: str
    group: str
    readOnlyHint: bool
    destructiveHint: bool
    idempotentHint: bool
    openWorldHint: bool
    approval_required: bool
    evidence_required: bool


CAPABILITY_RECORDS: tuple[CapabilityRecord, ...] = (
    CapabilityRecord("plan.mod", "planning", True, False, True, False, False, True),
    CapabilityRecord("plan.revise", "planning", True, False, True, False, False, True),
    CapabilityRecord("plan.approve", "planning", True, False, True, False, False, True),
    CapabilityRecord("evidence.search", "research", True, False, True, False, False, True),
    CapabilityRecord("project.inspect", "existing-project", True, False, True, False, False, True),
    CapabilityRecord("workflow.compile", "planning", True, False, True, False, False, True),
    CapabilityRecord("fabric.scaffold", "generation", False, False, False, False, True, True),
    CapabilityRecord("asset.generate", "generation", False, False, False, True, True, True),
    CapabilityRecord("world.ir.generate", "planning", False, False, False, False, False, True),
    CapabilityRecord("quality.validate", "quality", True, False, True, False, True, True),
    CapabilityRecord("build.gradle", "build", False, False, True, True, True, True),
    CapabilityRecord("test.gametest", "quality", False, False, True, False, True, True),
    CapabilityRecord("jar.inspect", "quality", True, False, True, False, False, True),
    CapabilityRecord("release.package", "release", False, False, True, False, True, True),
)


def capability_manifest() -> dict[str, Any]:
    return {
        "schema_version": "minecraft-mod-ai/capabilities-v3",
        "protocol_alignment": "Model Context Protocol; Python SDK FastMCP stdio server",
        "implementation_kind": "mcp-fastmcp-server-with-local-policy-broker",
        "server_entrypoint": "python -m minecraft_mod_ai.mcp_server",
        "authorization_source": "approved-proposal-hash-only",
        "retrieved_context_can_authorize": False,
        "tool_annotations_can_authorize": False,
        "runtime_mcp_1201": "disabled-until-version-compatible-fork-is-validated",
        "staged_discovery": {
            "planning": [
                "plan.mod",
                "plan.revise",
                "plan.approve",
                "evidence.search",
                "project.inspect",
                "workflow.compile",
                "world.ir.generate",
            ],
            "execution_after_approval": [
                "fabric.scaffold",
                "asset.generate",
                "quality.validate",
                "build.gradle",
                "test.gametest",
                "jar.inspect",
                "release.package",
            ],
        },
        "tools": [asdict(record) for record in CAPABILITY_RECORDS],
    }


def capability_manifest_hash() -> str:
    encoded = canonical_json(capability_manifest()).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def capability_names() -> frozenset[str]:
    return frozenset(record.name for record in CAPABILITY_RECORDS)
