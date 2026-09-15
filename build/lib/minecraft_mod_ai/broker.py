from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .capabilities import capability_manifest_hash, capability_names
from .spec import Proposal, ProposalStatus, SpecValidationError


class ToolAction(str, Enum):
    SCAFFOLD = "fabric.scaffold"
    VALIDATE = "quality.validate"
    GRADLE_BUILD = "build.gradle"
    GAME_TEST = "test.gametest"
    PACKAGE = "release.package"
    EXPORT = "release.package"


class PolicyDenied(PermissionError):
    pass


@dataclass(frozen=True)
class ToolRequest:
    action: ToolAction
    project_root: Path
    workspace_root: Path
    approved_hash: str


class LocalPolicyBroker:
    """Small default-deny broker for the local/Colab vertical slice."""

    _ALLOWED = frozenset(
        action for action in ToolAction if action.value in capability_names()
    )

    def authorize(self, request: ToolRequest, proposal: Proposal) -> None:
        if proposal.capability_manifest_hash != capability_manifest_hash():
            raise PolicyDenied(
                "Capability manifest drifted after approval; create and approve a new proposal."
            )
        if request.action not in self._ALLOWED:
            raise PolicyDenied(f"Tool action is not allowlisted: {request.action!r}")
        if proposal.status is not ProposalStatus.APPROVED:
            raise PolicyDenied("Write/build tools require an APPROVED proposal.")
        expected = proposal.calculate_hash()
        if request.approved_hash != expected:
            raise PolicyDenied("Tool request approval hash does not match the proposal.")
        workspace = request.workspace_root.resolve()
        project = request.project_root.resolve()
        try:
            project.relative_to(workspace)
        except ValueError as exc:
            raise PolicyDenied("Tool project_root escaped the approved workspace root.") from exc
        if project == workspace:
            raise PolicyDenied("Tools may not target the broad workspace root itself.")
        if project.is_symlink():
            raise PolicyDenied("Symlink project roots are not allowed.")


def approved_request(
    action: ToolAction,
    *,
    project_root: Path,
    workspace_root: Path,
    proposal: Proposal,
) -> ToolRequest:
    if proposal.status is not ProposalStatus.APPROVED:
        raise SpecValidationError("Proposal must be approved before creating a tool request.")
    return ToolRequest(
        action=action,
        project_root=project_root,
        workspace_root=workspace_root,
        approved_hash=proposal.calculate_hash(),
    )
