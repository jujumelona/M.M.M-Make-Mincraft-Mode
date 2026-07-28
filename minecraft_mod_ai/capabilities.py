from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .spec import canonical_json


@dataclass(frozen=True)
class CapabilityRecord:
    """Code-owned description of one broker capability.

    The four ``*Hint`` fields use the MCP 2025-11-25 tool-annotation vocabulary
    so that a future adapter can expose equivalent metadata.  This module is a
    local policy manifest; it does not implement or claim to be an MCP server.
    """

    name: str
    readOnlyHint: bool
    destructiveHint: bool
    idempotentHint: bool
    openWorldHint: bool
    approval_required: bool


CAPABILITY_RECORDS: tuple[CapabilityRecord, ...] = (
    CapabilityRecord(
        name="fabric.scaffold",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
        approval_required=True,
    ),
    CapabilityRecord(
        name="quality.validate",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
    ),
    CapabilityRecord(
        name="build.gradle",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
        approval_required=True,
    ),
    CapabilityRecord(
        name="test.gametest",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
    ),
    CapabilityRecord(
        name="release.package",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
        approval_required=True,
    ),
)


def capability_manifest() -> dict[str, Any]:
    """Return a fresh, deterministic manifest for the local policy broker."""

    return {
        "schema_version": "minecraft-mod-ai/capabilities-v1",
        "protocol_alignment": "MCP-2025-11-25 tool risk annotations",
        "implementation_kind": "local-policy-manifest-not-mcp-server",
        "authorization_source": "approved-proposal-hash-only",
        "retrieved_context_can_authorize": False,
        "tools": [asdict(record) for record in CAPABILITY_RECORDS],
    }


def capability_manifest_hash() -> str:
    encoded = canonical_json(capability_manifest()).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def capability_names() -> frozenset[str]:
    return frozenset(record.name for record in CAPABILITY_RECORDS)
