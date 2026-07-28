from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .spec import canonical_json


@dataclass(frozen=True)
class CapabilityRecord:
    """Code-owned description shared by the broker and MCP adapter.

    MCP annotations are discovery hints, never authorization.  The local broker
    independently checks proposal state, manifest hash and workspace scope.
    """

    name: str
    group: str
    readOnlyHint: bool
    destructiveHint: bool
    idempotentHint: bool
    openWorldHint: bool
    approval_required: bool
    evidence_required: bool


CAPABILITY_RECORDS: tuple[CapabilityRecord, ...] = (
    CapabilityRecord(
        name="plan.mod",
        group="planning",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=False,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="evidence.search",
        group="research",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=False,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="project.inspect",
        group="existing-project",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=False,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="workflow.compile",
        group="planning",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=False,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="fabric.scaffold",
        group="generation",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
        approval_required=True,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="quality.validate",
        group="quality",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="build.gradle",
        group="build",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
        approval_required=True,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="test.gametest",
        group="quality",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
        evidence_required=True,
    ),
    CapabilityRecord(
        name="release.package",
        group="release",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
        evidence_required=True,
    ),
)


def capability_manifest() -> dict[str, Any]:
    """Return a fresh, deterministic manifest for the local policy broker."""

    return {
        "schema_version": "minecraft-mod-ai/capabilities-v2",
        "protocol_alignment": "MCP Python SDK 2.0.0; compatible protocol negotiation",
        "implementation_kind": "local-policy-manifest-not-mcp-server",
        "authorization_source": "approved-proposal-hash-only",
        "retrieved_context_can_authorize": False,
        "tool_annotations_can_authorize": False,
        "staged_discovery": {
            "planning": ["plan.mod", "evidence.search", "project.inspect", "workflow.compile"],
            "execution_after_approval": [
                "fabric.scaffold",
                "quality.validate",
                "build.gradle",
                "test.gametest",
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
