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


def _cap(
    name: str,
    group: str,
    *,
    read: bool,
    destructive: bool = False,
    idempotent: bool = True,
    open_world: bool = False,
    approval: bool = False,
    evidence: bool = True,
) -> CapabilityRecord:
    return CapabilityRecord(
        name,
        group,
        read,
        destructive,
        idempotent,
        open_world,
        approval,
        evidence,
    )


CAPABILITY_RECORDS: tuple[CapabilityRecord, ...] = (
    _cap("plan.mod", "planning", read=True),
    _cap("plan.complete", "planning", read=True),
    _cap("plan.complete.approve", "planning", read=True),
    _cap("plan.revise", "planning", read=True),
    _cap("plan.approve", "planning", read=True),
    _cap("evidence.search", "research", read=True),
    _cap("evidence.index", "research", read=False),
    _cap("evidence.rerank", "research", read=True),
    _cap("project.inspect", "existing-project", read=True),
    _cap("workflow.compile", "planning", read=True),
    _cap("workflow.complete", "generation", read=False, idempotent=False, open_world=True, approval=True),
    _cap("source.patch", "generation", read=False, idempotent=False, approval=True),
    _cap("source.repair", "quality", read=False, idempotent=False, open_world=True, approval=True),
    _cap("fabric.scaffold", "generation", read=False, idempotent=False, approval=True),
    _cap("fabric.system.generate", "generation", read=False, idempotent=False, approval=True),
    _cap("asset.generate", "generation", read=False, idempotent=False, open_world=True, approval=True),
    _cap("audio.generate", "generation", read=False, idempotent=False, approval=True),
    _cap("blockbench.model", "generation", read=False, idempotent=False, open_world=True, approval=True),
    _cap("geckolib.generate", "generation", read=False, idempotent=False, approval=True),
    _cap("world.ir.generate", "planning", read=False, idempotent=False),
    _cap("world.compile", "generation", read=False, idempotent=False, approval=True),
    _cap("java.diagnostics", "quality", read=True),
    _cap("java.symbols", "quality", read=True),
    _cap("quality.validate", "quality", read=True, approval=True),
    _cap("build.gradle", "build", read=False, open_world=True, approval=True),
    _cap("test.gametest", "quality", read=False, approval=True),
    _cap("jar.inspect", "quality", read=True),
    _cap("runtime.instance", "runtime", read=False, idempotent=False, open_world=True, approval=True),
    _cap("runtime.command", "runtime", read=False, open_world=True, approval=True),
    _cap("runtime.screenshot", "runtime", read=True),
    _cap("mineflayer.playtest", "runtime", read=False, open_world=True, approval=True),
    _cap("model.smoke", "models", read=False, idempotent=False, open_world=True),
    _cap("training.trace.record", "training", read=False),
    _cap("training.dataset.export", "training", read=False),
    _cap("release.package", "release", read=False, approval=True),
    _cap("release.publish", "release", read=False, idempotent=False, open_world=True, approval=True),
)


def capability_manifest() -> dict[str, Any]:
    return {
        "schema_version": "minecraft-mod-ai/capabilities-v4",
        "protocol_alignment": "Model Context Protocol; Python SDK FastMCP stdio server",
        "implementation_kind": "mcp-fastmcp-server-with-local-policy-and-runtime-brokers",
        "server_entrypoint": "python -m minecraft_mod_ai.mcp_server",
        "authorization_source": "approved-proposal-hash-only-for-project-writes-and-runtime",
        "retrieved_context_can_authorize": False,
        "tool_annotations_can_authorize": False,
        "runtime_target": "disposable-minecraft-java-1.20.1-only",
        "staged_discovery": {
            "planning_research": [
                record.name
                for record in CAPABILITY_RECORDS
                if not record.approval_required
                and record.group in {"planning", "research", "existing-project", "models"}
            ],
            "generation_after_approval": [
                record.name
                for record in CAPABILITY_RECORDS
                if record.approval_required
                and record.group in {"generation", "build", "quality"}
            ],
            "runtime_after_build_and_approval": [
                record.name
                for record in CAPABILITY_RECORDS
                if record.group == "runtime"
            ],
            "training_from_verified_receipts_only": [
                "training.trace.record",
                "training.dataset.export",
            ],
        },
        "tools": [asdict(record) for record in CAPABILITY_RECORDS],
    }


def capability_manifest_hash() -> str:
    encoded = canonical_json(capability_manifest()).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def capability_names() -> frozenset[str]:
    return frozenset(record.name for record in CAPABILITY_RECORDS)
